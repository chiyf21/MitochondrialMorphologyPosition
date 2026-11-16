'''Extract Concat1152 and Project384 at the same native fusion forward.'''
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-gpu', required=True)
    parser.add_argument('--fish2-code', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--concat-dimension', type=int, choices=(768, 1152), default=1152)
    args = parser.parse_args()
    sys.path.insert(0, str(args.fish2_code.resolve()))
    from fish2_downstream.checkpoint import free_gpu
    from fish2_downstream.export_nohead_reviewed81 import ASSETS, reviewed_inputs, sha

    run = args.checkpoint.resolve().parents[1]
    launcher = json.loads((run / 'launcher.json').read_text())
    train = next(Path(arg) for arg in launcher['argv'] if arg.endswith('/refine/train.py'))
    revision = subprocess.check_output(['git', '-C', str(train.parent), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == launcher['code_commit']
    assert not subprocess.check_output(['git', '-C', str(train.parent), 'status', '--porcelain'])
    sys.path.insert(0, str(train.parent))
    from train import core, load_models

    free_gpu(args.physical_gpu)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config = state['config']
    completion = json.loads((run / 'complete.json').read_text())
    assert completion['status'] == 'training_complete' and completion['steps'] == state['step']
    assert state['stage'] == 'refine' and not config.get('linear_projection', False)
    no_balance = config.get('disable_pair_balance', False)
    target, groups, points, input_rows = reviewed_inputs(config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'input_rows.json').write_text(json.dumps(input_rows) + '\n')

    torch.set_num_threads(1)
    torch.manual_seed(20261111)
    torch.cuda.manual_seed_all(20261111)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    point, grouping, image = load_models(config, device)
    extra = {'image_mlp': True} if config.get('image_mlp', False) else {}
    model = core.System(point, image, disable_pair_balance=no_balance,
                        normalize_cls_regularizer=config.get('normalize_cls_regularizer', False), **extra).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    model.phase = 'refine'
    balance = {k: v.clone() for k, v in model.balance.state_dict().items()}

    concat = []
    # Fusion is invoked in Block 9–12 order. Capture the exact Block 12 MLP input.
    capture = model.fusion.mlp[0] if args.concat_dimension == 768 else model.fusion
    capture.register_forward_pre_hook(lambda module, inputs: concat.append(inputs[0]))

    @torch.inference_mode()
    def infer(batch, cache):
        concat.clear()
        inputs = core.prepare(batch, cache, point, grouping, image, device, 0, False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, fused, _ = model(inputs, views=1, loss=False)
        result = fused[12][:, 0].float().cpu().numpy()
        assert result.shape == (len(batch), 384) and np.isfinite(result).all()
        assert len(concat) == 4
        pair = concat[-1][:, 0].float().cpu().numpy()
        assert pair.shape == (len(batch), args.concat_dimension) and np.isfinite(pair).all()
        return pair, result

    qa = {}
    pairs = np.empty((len(target), args.concat_dimension), dtype=np.float32)
    result = np.empty((len(target), 384), dtype=np.float32)
    seen = np.zeros(len(target), dtype=bool)
    for source, rows in groups.items():
        sample = rows[:8]
        first = infer(sample, points[source])
        assert all(np.array_equal(a, b) for a, b in zip(first, infer(sample, points[source])))
        qa[source] = [row['record_id'] for row in sample]
        print('smoke passed', source, [x.shape for x in first], flush=True)
        if args.smoke:
            continue
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            indices = [row['embedding_row'] for row in batch]
            assert not seen[indices].any()
            pairs[indices], result[indices] = infer(batch, points[source])
            seen[indices] = True
            if start % 512 == 0 or start + 8 >= len(rows):
                print('exported', int(seen.sum()), 'of', len(target), flush=True)
    assert all(torch.equal(v, balance[k]) for k, v in model.balance.state_dict().items())
    if not args.smoke:
        assert seen.all()
        np.save(args.output / 'concat_12.npy', pairs)
        np.save(args.output / 'fused_12.npy', result)
        target.to_csv(args.output / 'instance_map.csv', index=False)
    manifest = dict(status='smoke_passed' if args.smoke else 'complete',
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                    runtime_commit=revision, runtime_train=str(train), task_assets=str(ASSETS),
                    epoch=state['epoch'], layer=12, dimensions=[args.concat_dimension, 384], pair_balance=not no_balance,
                    view='canonical, point sample 0, no augmentation, native FPS seed',
                    precision='FP32 preparation, BF16 model forward', batch_size=8,
                    smoke_record_ids=qa, exported=int(seen.sum()))
    (args.output / 'complete.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
