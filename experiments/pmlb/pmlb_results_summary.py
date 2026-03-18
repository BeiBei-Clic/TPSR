#!/usr/bin/env python3
"""
Summarize TPSR PMLB inference results by dataset group.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize TPSR PMLB results")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="experiments/pmlb/results/pmlb_results.csv",
        help="Path to the batch PMLB result CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="experiments/pmlb/results/pmlb_results_summary.csv",
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


def build_summary(input_df):
    summary_rows = []

    input_df = input_df.copy()
    input_df["group"] = input_df["dataset"].map(dataset_to_group)

    for group in GROUP_ORDER:
        group_df = input_df[input_df["group"] == group]
        total_count = int(len(group_df))

        raw_r2 = pd.to_numeric(group_df["r2"], errors="coerce")
        finite_r2 = raw_r2[np.isfinite(raw_r2)]
        r2_valid_count = int((finite_r2 >= 0).sum())

        cleaned_r2 = raw_r2.copy()
        cleaned_r2[~np.isfinite(cleaned_r2)] = 0.0
        cleaned_r2[cleaned_r2 < 0] = 0.0

        complexity_values = [
            value
            for value in (to_finite_float(v) for v in group_df["complexity"])
            if value is not None
        ]
        seconds_values = [
            value
            for value in (to_finite_float(v) for v in group_df["time"])
            if value is not None
        ]

        recovery_rate = 0.0
        if total_count > 0:
            recovery_hits = ((np.isfinite(raw_r2)) & (raw_r2 > 0.9)).sum()
            recovery_rate = float(recovery_hits / total_count)

        summary_rows.append(
            {
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

    return pd.DataFrame(summary_rows, columns=OUTPUT_COLUMNS)


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

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    input_df = pd.read_csv(args.input_csv)
    required_columns = {"dataset", "r2", "time", "complexity"}
    missing_columns = required_columns - set(input_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    summary_df = build_summary(input_df)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    print_summary_table(summary_df)
    print(f"\nSaved summary CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
