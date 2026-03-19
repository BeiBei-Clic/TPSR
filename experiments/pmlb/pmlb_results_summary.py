#!/usr/bin/env python3
"""Summarize TPSR PMLB inference results by dataset group and noise strength."""

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_CSV = "experiments/pmlb/results/pmlb_results_summary.csv"
DEFAULT_RESULT_DIR = Path("experiments/pmlb/results")
DEFAULT_INPUT_CANDIDATES = [
    DEFAULT_RESULT_DIR / "pmlb_batch_inference_noise_0.001.csv",
    DEFAULT_RESULT_DIR / "pmlb_batch_inference_noise_0.01.csv",
    DEFAULT_RESULT_DIR / "pmlb_batch_inference_noise_0.1.csv",
]
NOISE_ZERO_CANDIDATES = [
    DEFAULT_RESULT_DIR / "pmlb_batch_inference_noise_0.csv",
    DEFAULT_RESULT_DIR / "pmlb_results.csv",
]
OUTPUT_COLUMNS = [
    "noise_strength",
    "group",
    "r2_mean",
    "r2_var",
    "r2_valid_count",
    "total_count",
    "recovery_rate",
    "complexity_mean",
    "complexity_var",
    "complexity_count",
    "seconds_mean",
    "seconds_var",
    "seconds_count",
]
GROUP_ORDER = ["Feynman", "Strogatz", "Black-box"]
SUCCESS_STATUSES = {"ok", "success"}
NOISE_FILENAME_RE = re.compile(r"pmlb_batch_inference_noise_([^/]+)\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize TPSR PMLB results")
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        default=None,
        help="Paths to batch PMLB result CSV files. If omitted, auto-detect default files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the summary CSV",
    )
    return parser.parse_args()


def dataset_to_group(dataset_name):
    dataset_name = str(dataset_name)
    if dataset_name.startswith("feynman_"):
        return "Feynman"
    if dataset_name.startswith("strogatz_"):
        return "Strogatz"
    return "Black-box"


def resolve_input_csvs(input_csvs):
    if input_csvs:
        return [Path(path) for path in input_csvs]

    resolved_paths = []
    noise_zero_path = None
    for candidate in NOISE_ZERO_CANDIDATES:
        if candidate.exists():
            noise_zero_path = candidate
            break
    if noise_zero_path is None:
        missing = ", ".join(str(path) for path in NOISE_ZERO_CANDIDATES)
        raise FileNotFoundError(f"No zero-noise input CSV found. Tried: {missing}")
    resolved_paths.append(noise_zero_path)

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if not candidate.exists():
            raise FileNotFoundError(f"Input CSV not found: {candidate}")
        resolved_paths.append(candidate)

    return resolved_paths


def parse_noise_strength(path):
    path = Path(path)
    if path.name == "pmlb_results.csv":
        return 0.0

    match = NOISE_FILENAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer noise strength from file name: {path}")

    try:
        return float(match.group(1))
    except ValueError as exc:
        raise ValueError(f"Invalid noise strength in file name: {path}") from exc


def to_finite_float(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    return float(numeric)


def safe_mean(values):
    if not values:
        return np.nan
    return float(np.mean(values))


def safe_var(values):
    if not values:
        return np.nan
    return float(np.var(values, ddof=0))


def load_input_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    input_df = pd.read_csv(path)
    rename_map = {}
    if "time" in input_df.columns and "seconds" not in input_df.columns:
        rename_map["time"] = "seconds"
    if "expression" in input_df.columns and "expr" not in input_df.columns:
        rename_map["expression"] = "expr"
    if rename_map:
        input_df = input_df.rename(columns=rename_map)

    required_columns = {"dataset", "r2", "complexity", "seconds"}
    missing_columns = required_columns - set(input_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in input CSV {path}: {missing}")

    if "status" not in input_df.columns:
        input_df["status"] = "ok"

    input_df = input_df.copy()
    input_df["noise_strength"] = parse_noise_strength(path)
    return input_df


def build_summary(input_df):
    summary_rows = []

    input_df = input_df.copy()
    input_df["group"] = input_df["dataset"].map(dataset_to_group)

    noise_levels = sorted(input_df["noise_strength"].drop_duplicates().tolist())
    for noise_strength in noise_levels:
        noise_df = input_df[input_df["noise_strength"] == noise_strength]
        for group in GROUP_ORDER:
            group_df = noise_df[noise_df["group"] == group]
            total_count = int(len(group_df))

            raw_r2 = pd.to_numeric(group_df["r2"], errors="coerce")
            finite_r2 = raw_r2[np.isfinite(raw_r2)]
            r2_valid_count = int((finite_r2 >= 0).sum())

            cleaned_r2 = raw_r2.copy()
            cleaned_r2[~np.isfinite(cleaned_r2)] = 0.0
            cleaned_r2[cleaned_r2 < 0] = 0.0

            status_series = group_df["status"].astype(str).str.lower()
            success_mask = status_series.isin(SUCCESS_STATUSES)
            success_df = group_df[success_mask]

            complexity_values = [
                value
                for value in (to_finite_float(v) for v in success_df["complexity"])
                if value is not None
            ]
            seconds_values = [
                value
                for value in (to_finite_float(v) for v in success_df["seconds"])
                if value is not None
            ]

            recovery_rate = 0.0
            if total_count > 0:
                recovery_hits = int(((np.isfinite(raw_r2)) & (raw_r2 > 0.9)).sum())
                recovery_rate = float(recovery_hits / total_count)

            summary_rows.append(
                {
                    "noise_strength": float(noise_strength),
                    "group": group,
                    "r2_mean": safe_mean(cleaned_r2.tolist()),
                    "r2_var": safe_var(cleaned_r2.tolist()),
                    "r2_valid_count": r2_valid_count,
                    "total_count": total_count,
                    "recovery_rate": recovery_rate,
                    "complexity_mean": safe_mean(complexity_values),
                    "complexity_var": safe_var(complexity_values),
                    "complexity_count": int(len(complexity_values)),
                    "seconds_mean": safe_mean(seconds_values),
                    "seconds_var": safe_var(seconds_values),
                    "seconds_count": int(len(seconds_values)),
                }
            )

    summary_df = pd.DataFrame(summary_rows, columns=OUTPUT_COLUMNS)
    summary_df = summary_df.sort_values(
        by=["noise_strength", "group"],
        key=lambda series: series.map({name: idx for idx, name in enumerate(GROUP_ORDER)})
        if series.name == "group"
        else series,
    )
    return summary_df.reset_index(drop=True)


def format_cell(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def print_summary_table(summary_df):
    display_df = summary_df.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].map(format_cell)

    widths = {
        column: max(len(column), display_df[column].map(len).max())
        for column in display_df.columns
    }

    header = " | ".join(column.ljust(widths[column]) for column in display_df.columns)
    separator = "-+-".join("-" * widths[column] for column in display_df.columns)

    print("\nPMLB results summary:")
    print(header)
    print(separator)
    for _, row in display_df.iterrows():
        print(" | ".join(row[column].ljust(widths[column]) for column in display_df.columns))


def main():
    args = parse_args()
    input_paths = resolve_input_csvs(args.input_csvs)
    input_frames = [load_input_csv(path) for path in input_paths]
    input_df = pd.concat(input_frames, ignore_index=True)
    summary_df = build_summary(input_df)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    print_summary_table(summary_df)
    print(f"\nSaved summary CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
