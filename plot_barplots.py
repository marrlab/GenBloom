#!/usr/bin/env python
"""Regenerate the visual downstream bar plots from released metric CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSIFICATION_DATASETS = ["aml_hehr", "apl_aml", "catiomorph_binary_fold"]
RETRIEVAL_DATASETS = ["aml_hehr", "apl_aml", "catiomorph_binary"]
CLASSIFICATION_METHODS = ["knn_k_5", "logistic_regression"]

DATASET_LABELS = {
    "aml_hehr": "AML-Hehr",
    "apl_aml": "APL-AML",
    "catiomorph_binary_fold": "cAItomorph\nbinary",
    "catiomorph_binary": "cAItomorph\nbinary",
}
METHOD_LABELS = {
    "knn_k_5": "KNN (k=5)",
    "logistic_regression": "Log. Reg.",
}
MODEL_ORDER = ["TITAN", "PRISM", "GigaPath", "dinoBloom Mean", "GenBloom-V", "GenBloom-G"]
RETRIEVAL_MODEL_ORDER = ["Mean Pool", "TITAN", "PRISM", "GigaPath", "GenBloom-V", "GenBloom-G"]
COLORS = {
    "Mean Pool": "#7f7f7f",
    "TITAN": "#1f77b4",
    "PRISM": "#d62728",
    "GigaPath": "#ff7f0e",
    "dinoBloom Mean": "#8c564b",
    "GenBloom-V": "#2ca02c",
    "GenBloom-G": "#9467bd",
}


def _read_classification_csv(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = model
    df["dataset"] = df["dataset"].astype(str).str.lower().replace(
        {"beluga": "caitomorph"}
    )
    return df


def load_classification(reference_dir: Path) -> pd.DataFrame:
    class_dir = reference_dir / "classification"
    frames = []
    for model, rel in [
        ("TITAN", "titan/all_metrics.csv"),
        ("PRISM", "prism/all_metrics.csv"),
        ("GigaPath", "gigapath/all_metrics.csv"),
        ("dinoBloom Mean", "dinobloom_mean/all_metrics.csv"),
        ("GenBloom-V", "genbloom_v/all_metrics.csv"),
    ]:
        path = class_dir / rel
        if path.exists():
            frames.append(_read_classification_csv(path, model))

    for path in sorted((class_dir / "genbloom_g").glob("fold_*/all_metrics.csv")):
        frames.append(_read_classification_csv(path, "GenBloom-G"))

    if not frames:
        raise FileNotFoundError(f"No classification metrics found under {class_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df[
        df["dataset"].isin(CLASSIFICATION_DATASETS)
        & df["method"].isin(CLASSIFICATION_METHODS)
        & (df["split"] == "test")
    ].copy()
    df["fold"] = df["fold"].astype(str)

    multi = df[df["fold"] != "single"].copy()
    multi["fold"] = pd.to_numeric(multi["fold"], errors="coerce")
    multi = (
        multi.groupby(["model", "dataset", "method", "fold"], as_index=False)[
            "balanced_accuracy"
        ]
        .mean()
        .dropna(subset=["fold"])
    )
    multi = (
        multi.groupby(["model", "dataset", "method"])["balanced_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    single_raw = df[df["fold"] == "single"].copy()
    parts = [multi]
    if not single_raw.empty:
        single = (
            single_raw.groupby(["model", "dataset", "method"])["balanced_accuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )
        parts.append(single)

    agg = pd.concat(parts, ignore_index=True)
    agg = agg.rename(columns={"mean": "balanced_accuracy_mean", "std": "balanced_accuracy_std"})
    agg["balanced_accuracy_std"] = agg["balanced_accuracy_std"].fillna(0.0)
    return agg


def plot_classification(values: pd.DataFrame, output_dir: Path) -> None:
    groups = [(dataset, method) for dataset in CLASSIFICATION_DATASETS for method in CLASSIFICATION_METHODS]
    labels = [f"{DATASET_LABELS[d]}\n{METHOD_LABELS[m]}" for d, m in groups]
    models = [m for m in MODEL_ORDER if m in set(values["model"])]

    x = np.arange(len(groups))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for idx, model in enumerate(models):
        means, stds = [], []
        for dataset, method in groups:
            row = values[
                (values["model"] == model)
                & (values["dataset"] == dataset)
                & (values["method"] == method)
            ]
            means.append(float(row["balanced_accuracy_mean"].iloc[0]) if len(row) else np.nan)
            stds.append(float(row["balanced_accuracy_std"].iloc[0]) if len(row) else 0.0)
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=model,
            color=COLORS.get(model, "#999999"),
        )

    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "classification_barplot.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "classification_barplot.pdf", bbox_inches="tight")
    plt.close(fig)


def load_retrieval(reference_dir: Path) -> pd.DataFrame:
    retrieval_dir = reference_dir / "patient_retrieval"
    frames = []
    for model, rel in [
        ("Mean Pool", "mean_pool/all_bootstrap_retrieval.csv"),
        ("TITAN", "titan/all_bootstrap_retrieval.csv"),
        ("PRISM", "prism/all_bootstrap_retrieval.csv"),
        ("GigaPath", "gigapath/all_bootstrap_retrieval.csv"),
        ("GenBloom-V", "genbloom_v/all_bootstrap_retrieval.csv"),
        ("GenBloom-G", "genbloom_g/all_bootstrap_retrieval.csv"),
    ]:
        path = retrieval_dir / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No patient retrieval metrics found under {retrieval_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = df["dataset"].replace({"catiomorph_binary_fold": "catiomorph_binary"})
    return df[
        df["dataset"].isin(RETRIEVAL_DATASETS)
        & (df["metric"] == "test_to_test/map_at_k/@3")
    ].copy()


def plot_retrieval(values: pd.DataFrame, output_dir: Path) -> None:
    models = [m for m in RETRIEVAL_MODEL_ORDER if m in set(values["model"])]
    x = np.arange(len(RETRIEVAL_DATASETS))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for idx, model in enumerate(models):
        means, stds = [], []
        for dataset in RETRIEVAL_DATASETS:
            row = values[(values["model"] == model) & (values["dataset"] == dataset)]
            means.append(float(row["model_mean"].iloc[0]) if len(row) else np.nan)
            stds.append(float(row["model_std"].iloc[0]) if len(row) else 0.0)
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=model,
            color=COLORS.get(model, "#999999"),
        )

    ax.set_ylabel("mAP@3")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in RETRIEVAL_DATASETS])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "patient_retrieval_map3.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "patient_retrieval_map3.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("outputs/classification"),
        help="Directory containing classification outputs (outputs/classification/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Where to write plots and plotted-value CSVs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    classification = load_classification(args.reference_dir)
    retrieval = load_retrieval(args.reference_dir)

    classification.to_csv(args.output_dir / "classification_barplot_values.csv", index=False)
    retrieval.to_csv(args.output_dir / "patient_retrieval_map3_values.csv", index=False)
    plot_classification(classification, args.output_dir)
    plot_retrieval(retrieval, args.output_dir)

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
