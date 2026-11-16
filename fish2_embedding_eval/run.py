'''Compare completed Fish2 runs at Concat1152 and Project384.'''
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP


STAGES = {'concat': 1152, 'project': 384}
UMAP_ARGS = dict(n_neighbors=30, min_dist=0.1, metric='cosine',
                 random_state=20261116, n_jobs=1)
REFINE_WEIGHTS = dict(invariance=25, variance=5, covariance=0.2, infonce=1,
                      point_variance=1.25, point_covariance=0.05,
                      image_variance=1.25, image_covariance=0.05)


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def epoch_losses(path, final_epoch):
    groups = {}
    duplicate_count = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['phase'], row['epoch']
            batch = row['batch']
            steps = groups.setdefault(key, {})
            if batch in steps:
                duplicate_count[key] = duplicate_count.get(key, 0) + 1
            steps[batch] = row
    results = []
    for key, steps in sorted(groups.items()):
        phase, epoch = key
        rows = [steps[k] for k in sorted(steps)]
        if set(steps) != set(range(1, rows[0]['batches_per_epoch'] + 1)):
            raise ValueError(f'Incomplete batches at {path} {key}')
        if len({r['batches_per_epoch'] for r in rows}) != 1:
            raise ValueError(f'Batch count changed at {path} {key}')
        counters = [r['optimizer_step'] for r in rows]
        if counters != list(range(counters[0], counters[0] + len(rows))):
            raise ValueError(f'Non-contiguous optimizer steps at {path} {key}')
        result = dict(phase=phase, epoch=epoch, steps=len(rows),
                      optimizer_step=counters[-1], duplicate_log_rows=duplicate_count.get(key, 0))
        result['total'] = float(np.mean([row['loss'] for row in rows]))
        if phase == 'refine':
            for part, weight in REFINE_WEIGHTS.items():
                result[part] = float(np.mean([r['parts'][f'{layer}/{part}']
                                              for r in rows for layer in range(9, 13)]))
                result[f'weighted_{part}'] = weight * result[part]
            reconstructed = sum(result[f'weighted_{k}'] for k in REFINE_WEIGHTS)
            if not math.isclose(reconstructed, result['total'], rel_tol=2e-6, abs_tol=1e-5):
                raise ValueError(f'Logged loss disagrees with configured formula: {path} {key}')
        elif phase == 'pretrain':
            for layer in range(9, 13):
                result[f'nna_block_{layer}'] = float(np.mean([r['parts'][str(layer)] for r in rows]))
        else:
            raise ValueError(f'Unsupported phase {phase}')
        results.append(result)
    if [r['epoch'] for r in results if r['phase'] == 'refine'] != list(range(1, final_epoch + 1)):
        raise ValueError(f'Missing refinement epochs at {path}')
    return results


def downstream(entry, descriptor, name, output, assets):
    def call(*args):
        return subprocess.check_output([sys.executable, str(entry), *map(str, args)], text=True,
                                       stderr=subprocess.STDOUT)
    (output / f'{name}_inspect.json').write_text(call('inspect', '--embedding', descriptor,
                                                     '--assets', assets))
    log = output / f'{name}_downstream.log'
    with log.open('w') as stream:
        subprocess.run([sys.executable, str(entry), 'run', '--embedding', str(descriptor),
                        '--assets', str(assets), '--task', 'all', '--classifier', 'lr',
                        '--owner', 'codex', '--name', name, '--workers', '2'],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    run = Path(next(line.removeprefix('COMPLETE ') for line in log.read_text().splitlines()
                    if line.startswith('COMPLETE ')))
    if not (run / '_SUCCESS.json').is_file():
        raise ValueError(f'Downstream run incomplete: {run}')
    return run


def aligned_embedding(descriptor, target, dimension, checkpoint):
    spec = read(descriptor)
    if Path(spec['checkpoint']['path']).resolve() != checkpoint:
        raise ValueError(f'Checkpoint mismatch: {descriptor}')
    index = pd.read_csv(spec['instance_map'], dtype={'record_id': str, 'body_id': str})
    index = index.set_index('record_id', verify_integrity=True).loc[target.index]
    if not index.body_id.equals(target.body_id):
        raise ValueError(f'Body mismatch: {descriptor}')
    matrix = np.load(spec['matrix'], allow_pickle=False)
    x = matrix[index.embedding_row.to_numpy(dtype=int)]
    if x.shape != (len(target), dimension) or not np.isfinite(x).all():
        raise ValueError(f'Invalid shape or values: {descriptor}')
    return x


def umap_plot(panels, target, label, path):
    classes = sorted(target[label].dropna().unique())
    if label == 'compartment':
        classes = ['axon', 'dendrite']
    fig, axes = plt.subplots(len(panels) // 2, 2, squeeze=False,
                             figsize=(10, 4 * (len(panels) // 2)), layout='constrained')
    for ax, (title, coords) in zip(axes.flat, panels.items()):
        valid = target[label].isin(classes).to_numpy()
        if label == 'compartment':
            valid = valid & target['compartment_qc'].eq('ok').to_numpy()
        if (~valid).any():
            ax.scatter(*coords[~valid].T, c='lightgray', s=2, linewidths=0, label='Outside task')
        for k, value in enumerate(classes):
            keep = valid & target[label].eq(value).to_numpy()
            ax.scatter(*coords[keep].T, color=plt.get_cmap('tab10')(k), s=3,
                       linewidths=0, alpha=0.65, label=value)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set(xticks=[], yticks=[])
    axes.flat[0].legend(markerscale=3, fontsize=7, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def finish(output, panels, target, provenance):
    for label in ('cell_type', 'compartment'):
        umap_plot(panels, target, label, output / f'umap_{label}.png')
    loss_frame = pd.read_csv(output / 'epoch_loss.csv').loc[lambda frame: frame.phase.eq('refine')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for name, group in loss_frame.groupby('run', sort=False):
        shared = group[[f'weighted_{k}' for k in ('invariance', 'variance', 'covariance', 'infonce')]]
        axes[0].plot(group.epoch, group.total, label=name)
        axes[1].plot(group.epoch, shared.sum(axis=1), label=name)
    axes[0].set(title='Total training loss', yscale='log', xlabel='Refine epoch')
    axes[1].set(title='Common fused VICReg + InfoNCE terms', xlabel='Refine epoch')
    for ax in axes:
        ax.legend(frameon=False)
    fig.savefig(output / 'loss_curves.png', dpi=180)
    plt.close(fig)
    (output / 'complete.json').write_text(json.dumps(dict(runs=provenance,
        umap=UMAP_ARGS, pca_components=30, instances=len(target)), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    assets = Path(cfg['assets'])
    entry = Path(cfg['fish2_entry'])
    target = pd.read_csv(assets / 'instances.csv', dtype={'record_id': str, 'body_id': str})
    target = target.set_index('record_id', verify_integrity=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(args.config.read_text())
    all_metrics, losses, panels = [], [], {}
    provenance = {}
    for item in cfg['runs']:
        name = item['name']
        ck = Path(item['checkpoint']).resolve(strict=True)
        run = ck.parents[1]
        stamp = read(run / 'complete.json')
        e = int(ck.stem.removeprefix('refine_epoch'))
        if not ck.name.startswith('refine_epoch') or stamp['status'] != 'training_complete':
            raise ValueError(f'Expected completed refine checkpoint: {ck}')
        rows = epoch_losses(run / 'step_metrics.jsonl', e)
        if rows[-1]['optimizer_step'] != stamp['steps']:
            raise ValueError(f'Checkpoint is not the final training endpoint: {ck}')
        losses += [dict(run=name, **r) for r in rows]
        provenance[name] = dict(checkpoint=str(ck), epoch=e, loss_profile=item['loss_profile'],
                                pair_balance=item['balance'], representations={})
        for stage, dim in STAGES.items():
            descriptor = Path(item[stage])
            x = aligned_embedding(descriptor, target, dim, ck)
            key = f'{name}_{stage}'
            pc = PCA(n_components=30, svd_solver='randomized', random_state=UMAP_ARGS['random_state'])
            coords = UMAP(**UMAP_ARGS).fit_transform(pc.fit_transform(x))
            write_csv(output / f'{key}_umap.csv',
                      (dict(record_id=i, umap_x=u, umap_y=v)
                       for i, (u, v) in zip(target.index, coords)))
            panels[f'{name}, E{e} — {stage.title()}{dim}'] = coords
            evaluation = downstream(entry, descriptor, cfg['name'] + '_' + key, output, assets)
            provenance[name]['representations'][stage] = dict(downstream=str(evaluation),
                descriptor=str(descriptor), pca30_variance=float(pc.explained_variance_ratio_.sum()))
            for task, values in read(evaluation / 'metrics.json').items():
                for unit in ('per_mito', 'per_group'):
                    metrics = values['embedding'][unit]
                    all_metrics.append(dict(run=name, epoch=e, stage=stage, dimension=dim,
                        task=task, unit=unit, balanced_accuracy=metrics['balanced_accuracy'],
                        macro_f1=metrics['macro_f1']))
            print(name, stage, evaluation, flush=True)
    write_csv(output / 'epoch_loss.csv', losses)
    write_csv(output / 'downstream_metrics.csv', all_metrics)
    finish(output, panels, target, provenance)


if __name__ == '__main__':
    main()
