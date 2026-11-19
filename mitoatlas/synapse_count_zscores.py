"""Summarize endpoint counts and express saved Ridge predictions in training-fold SDs."""
import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from fish2_downstream.common import dump, sha


MODELS = ("train_mean", "geometry", "21D_only", "concat_only", "geometry_21D", "geometry_concat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    distributions, summaries, normalizers = [], [], []
    hashes = {}
    for scale in ("compartment", "bin10um"):
        for role in ("pre", "post"):
            path = args.run / f"oof_{scale}_pooled_{role}_ridge.csv"
            hashes[path.name] = sha(path)
            data = pd.read_csv(path, dtype={"body_id": str})
            obs = f"{role}_count"
            fields = [obs, *MODELS]
            fold = data.cell_type_fold
            if (data.duplicated(["body_id","unit"]).any() or set(fold) != set(range(5))
                    or not data.groupby("body_id").cell_type_fold.nunique().eq(1).all()
                    or not np.isfinite(data[fields].to_numpy()).all()):
                raise ValueError(f"Invalid held-out prediction table: {path}")

            groups = [("pooled", data), *data.groupby("compartment", sort=True)]
            for group, subset in groups:
                v = subset[obs]
                q = v.quantile([.25,.5,.75,.90,.95])
                length_q = subset.length_um.quantile([.25,.5,.75])
                distributions.append(dict(scale=scale, target=role, compartment=group, units=len(v),
                    mean=v.mean(), std=v.std(ddof=0), q25=q[.25], median=q[.5], q75=q[.75], p90=q[.90],
                    p95=q[.95], max=v.max(), zero_fraction=v.eq(0).mean(),
                    nonzero_median=v[v.gt(0)].median(), length_q25_um=length_q[.25],
                    length_median_um=length_q[.5], length_q75_um=length_q[.75]))

            standardized = pd.DataFrame(index=data.index, columns=fields, dtype=float)
            for f in range(5):
                heldout = fold.eq(f)
                reference = data.loc[~heldout,obs]
                mean, std = reference.mean(), reference.std(ddof=0)
                if std <= 0 or not np.isclose(data.loc[heldout,"train_mean"],mean).all():
                    raise ValueError(f"Invalid training mean or SD in fold {f} of {path}")
                standardized.loc[heldout,fields] = (data.loc[heldout,fields] - mean) / std
                normalizers.append(dict(scale=scale, target=role, fold=f, training_units=len(reference),
                                        mean_count=mean, sd_count=std))
            for model in MODELS:
                raw = data[fields]
                per_fold = [
                    r2_score(standardized.loc[fold.eq(f),obs], standardized.loc[fold.eq(f), model])
                    for f in range(5)
                ]
                raw_r2 = [
                    r2_score(raw.loc[fold.eq(f),obs], raw.loc[fold.eq(f), model])
                    for f in range(5)
                ]
                np.testing.assert_allclose(per_fold, raw_r2, atol=1e-12)
                summaries.append(dict(scale=scale, target=role, model=model, units=len(data),
                    pooled_count_r2=r2_score(raw[obs], raw[model]),
                    pooled_z_r2=r2_score(standardized[obs], standardized[model]),
                    fold_mean_z_r2=np.mean(per_fold), mae_sd=mean_absolute_error(standardized[obs], standardized[model])))
            pd.concat([data, standardized.add_suffix("_z")], axis=1).to_csv(
                args.output / f"oof_{scale}_pooled_{role}_z.csv", index=False)
    for filename, rows in (("count_distributions.csv", distributions), ("z_scores.csv", summaries),
                           ("fold_normalizers.csv", normalizers)):
        pd.DataFrame(rows).to_csv(args.output / filename, index=False)
    dump(args.output / "run.json", dict(source=str(args.run), input_sha256=hashes,
        source_run_sha256=sha(args.run / "run.json"),
        commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        normalization="For each fold and target, subtract the training mean and divide by training population SD."))


if __name__ == "__main__":
    main()
