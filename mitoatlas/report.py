"""Write a per-experiment Markdown summary from an existing MitoAtlas run."""
import argparse
from pathlib import Path

import pandas as pd

from fish2_downstream.common import read

TITLES = {
    "R1": "Neuron type classification",
    "R2": "Compartment classification",
    "R3": "Morphological information probe",
    "R4": "Morphological distribution differences",
    "R5": "Cross-compartment coordination",
    "A1": "Morphological state atlas",
    "A2": "Geometry-matched EM candidates",
    "A3": "Incremental prediction of independent texture",
}


def markdown_table(rows, headers):
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "&#124;").replace("\n", " ")

    def row(items):
        return "| " + " | ".join(map(cell, items)) + " |"

    return [row(headers), row(["---"] * len(headers)), *(row(items) for items in rows)]


def classification(directory):
    rows, links = [], []
    for path in sorted(directory.glob("*/metrics.json")):
        metrics = read(path)
        for level, label in (("per_mito", "Instance"), ("per_group", "Group")):
            if level not in metrics:
                continue
            values = metrics[level]
            rows.append((path.parent.name, label, values["balanced_accuracy"], values["macro_f1"]))
        prefix = path.parent.name
        links.append(f"- {prefix}: [metrics and confusion matrices]({prefix}/metrics.json), "
                     f"[instance predictions]({prefix}/instance_oof.csv), "
                     f"[group predictions]({prefix}/group_oof.csv)")
    if directory.name == "R2":
        links.append("Group scores use annotated neuron × compartment groups.")
    return markdown_table(rows, ["Representation and classifier", "Level", "Balanced accuracy", "Macro F1"]) + ["", *links]


def regression(directory):
    frame = pd.read_csv(directory / "scores.csv")
    complete = frame[frame.status.eq("complete")]
    if complete.empty:
        lines = ["No nonconstant targets were evaluated."]
    elif directory.name == "R3":
        summary = complete.groupby("predictors").r2.agg(["size", "median"]).reset_index()
        lines = markdown_table(summary.itertuples(index=False, name=None),
                               ["Representation", "Targets", "Median held-out R²"])
    else:
        cols = ["target", "predictors", "r2", "mae", "delta_r2_vs_geometry", "delta_mae_vs_geometry"]
        lines = markdown_table(complete[cols].itertuples(index=False, name=None),
                               ["Texture", "Predictors", "R²", "MAE", "Δ R²", "Δ MAE"])
    lines += ["", "[Scores by target](scores.csv)"]
    if (directory / "oof.csv").exists():
        lines[-1] += " · [Held-out predictions and residuals](oof.csv)"
    return lines


def morphology(directory):
    frame = pd.read_csv(directory / "results.csv")
    if directory.name == "R4":
        grouped = frame.groupby(["representation", "contrast"]).distance
        headers = ["Representation", "Contrast", "Comparisons", "Median distance"]
        note = "Raw distance depends on representation dimension; interpret each space separately."
    else:
        grouped = frame.groupby("representation").cosine
        headers = ["Representation", "Defined type pairs", "Median cosine"]
        note = "The full table contains each type pair and its neuron bootstrap interval."
    summary = grouped.agg(["count", "median"]).reset_index()
    return markdown_table(summary.itertuples(index=False, name=None), headers) + [
        "", note, "", "[Results by contrast](results.csv)",
    ]


def atlas(directory):
    rows, links = [], []
    for spectrum_path in sorted(directory.glob("*/spectrum.csv")):
        rep = spectrum_path.parent.name
        spectrum = pd.read_csv(spectrum_path)
        stability = pd.read_csv(spectrum_path.with_name("stability.csv"))
        for k, values in stability.groupby("k"):
            rows.append((rep, spectrum.effective_rank.iloc[0], k, values.ari.median()))
        if stability.empty:
            rows.append((rep, spectrum.effective_rank.iloc[0], None, None))
        files = [("PCA spectrum", "spectrum.csv"), ("states", "states.csv"),
                 ("stability", "stability.csv"), ("gallery IDs", "gallery_ids.csv"),
                 ("composition", "composition.csv"), ("morphometrics", "cluster_morphometrics.csv")]
        links.append(f"- {rep}: " + " · ".join(f"[{label}]({rep}/{name})" for label, name in files
                                             if (spectrum_path.parent / name).exists()))
    return markdown_table(rows, ["Representation", "Effective rank", "Clusters", "Median seed ARI"]) + [
        "", "Candidate partitions are descriptive.", "", *links,
    ]


def candidates(directory):
    frame = pd.read_csv(directory / "candidate_pairs.csv", dtype={"record_a": str, "record_b": str})
    return [
        f"{len(frame):,} directed candidate pairs from {frame.record_a.nunique():,} anchors.",
        "",
        "[Candidate pairs and learned distance rankings](candidate_pairs.csv)",
        "",
        "Candidates prioritize EM review. They do not establish ultrastructural differences.",
    ]


SUMMARIES = dict(R1=classification, R2=classification, R3=regression, R4=morphology,
                 R5=morphology, A1=atlas, A2=candidates, A3=regression)


def write_report(run_dir):
    run = read(run_dir / "run.json")
    lines = ["# MitoAtlas results", "", f"Dataset: {run['arguments']['dataset']}",
             f"Input cohort: {run['instances']:,} instances, {run['bodies']:,} neurons",
             f"Code revision: `{run['commit']}`", "", "[Run provenance](run.json)", ""]
    status = []
    for task, record in run["tasks"].items():
        status.append((f"[{task}](./{task}/report.md)", TITLES[task], record["status"],
                       record.get("instances"), record.get("bodies")))
        section = [f"# {task} — {TITLES[task]}", "", f"Status: **{record['status']}**", ""]
        if record["status"] == "COMPLETE":
            section += [f"Cohort: {record['instances']:,} instances, {record['bodies']:,} neurons.",
                        "", "[Exact instance list](cohort.csv)", "", *SUMMARIES[task](run_dir / task)]
        else:
            section.append(record.get("reason", "No result available."))
        (run_dir / task / "report.md").write_text("\n".join(section) + "\n", encoding="utf-8")
    lines += markdown_table(status, ["Experiment", "Question", "Status", "Instances", "Neurons"])
    output = run_dir / "report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    print(write_report(parser.parse_args().run_dir))


if __name__ == "__main__":
    main()
