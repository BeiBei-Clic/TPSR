#!/usr/bin/env python3
"""Simplify existing TPSR PMLB result expressions with SymPy."""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.pmlb.pmlb_results_summary import parse_noise_strength
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.metrics import compute_metrics


SUCCESS_STATUSES = {"success"}
UNARY_FUNCTIONS = {
    "abs": sp.Abs,
    "arctan": sp.atan,
    "cos": sp.cos,
    "exp": sp.exp,
    "inv": lambda value: 1 / value,
    "log": sp.log,
    "sin": sp.sin,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
}
BINARY_OPERATORS = {"add", "sub", "mul", "div", "pow"}
ROW_TIMEOUT_SECONDS = 15


def raise_timeout(signum, frame):
    raise TimeoutError


def parse_args():
    parser = argparse.ArgumentParser(description="Simplify TPSR PMLB result CSVs with SymPy")
    parser.add_argument(
        "--input_csvs",
        nargs="+",
        required=True,
        help="Existing TPSR batch result CSV files",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="pmlb/datasets",
        help="Path to the local PMLB datasets directory",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional row cap, kept consistent with the inference script",
    )
    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="Training-point cap used during the original inference split",
    )
    return parser.parse_args()


def get_default_params():
    parser = get_parser()
    params, _ = parser.parse_known_args([])
    params.lam = 0.1
    params.width = 3
    params.rollout = 3
    params.num_beams = 1
    params.horizon = 200
    params.no_seq_cache = False
    params.no_prefix_cache = True
    params.max_input_points = 200
    params.rescale = True
    params.max_input_dimension = 10
    params.n_trees_to_refine = 10
    params.max_number_bags = 1
    params.device = "cpu"
    return params


def read_file(filename, label="target", sep=None):
    compression = "gzip" if filename.endswith("gz") else None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression, engine="python")
    feature_names = np.array([column for column in input_data.columns.values if column != label])
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]
    return X, y, feature_names


def select_training_data(X, y, max_rows, max_input_points):
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max_rows must be positive when provided.")
        X = X[:max_rows]
        y = y[:max_rows]

    x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=29910
    )

    if max_input_points <= 0:
        raise ValueError("--max_input_points must be positive.")
    if len(x_to_fit) > max_input_points:
        x_to_fit = x_to_fit[:max_input_points]
        y_to_fit = y_to_fit[:max_input_points]

    return x_to_fit, x_to_predict, y_to_fit, y_to_predict


def tokenize_expression(expr_text):
    tokens = []
    index = 0
    while index < len(expr_text):
        char = expr_text[index]
        if char.isspace():
            index += 1
            continue
        if expr_text.startswith("**", index):
            tokens.append("**")
            index += 2
            continue
        if char in "()":
            tokens.append(char)
            index += 1
            continue
        if char in "+-" or char.isdigit() or char == ".":
            start = index
            if char in "+-":
                index += 1
            while index < len(expr_text) and (expr_text[index].isdigit() or expr_text[index] == "."):
                index += 1
            if index < len(expr_text) and expr_text[index] in "eE":
                index += 1
                if index < len(expr_text) and expr_text[index] in "+-":
                    index += 1
                while index < len(expr_text) and expr_text[index].isdigit():
                    index += 1
            tokens.append(expr_text[start:index])
            continue
        if char.isalpha() or char == "_":
            start = index
            while index < len(expr_text) and (expr_text[index].isalnum() or expr_text[index] == "_"):
                index += 1
            tokens.append(expr_text[start:index])
            continue
        raise ValueError(f"Unsupported character '{char}' in expression: {expr_text}")
    return tokens


def parse_number_token(token):
    if any(marker in token for marker in (".", "e", "E")):
        return sp.Float(token)
    return sp.Integer(token)


def parse_atom(tokens, position):
    token = tokens[position]
    if token.startswith("x_"):
        return sp.Symbol(token, real=True), position + 1
    if token == "e":
        return sp.E, position + 1
    if token == "pi":
        return sp.pi, position + 1
    if token == "euler_gamma":
        return sp.EulerGamma, position + 1
    if token == "inf":
        return sp.oo, position + 1
    if token == "nan":
        return sp.nan, position + 1
    if token[0] in "+-0123456789.":
        return parse_number_token(token), position + 1
    raise ValueError(f"Unsupported token '{token}'")


def apply_binary_operator(operator_name, left_value, right_value):
    if operator_name == "add":
        return left_value + right_value
    if operator_name == "sub":
        return left_value - right_value
    if operator_name == "mul":
        return left_value * right_value
    if operator_name == "div":
        return left_value / right_value
    if operator_name == "pow":
        return left_value ** right_value
    raise ValueError(f"Unsupported binary operator '{operator_name}'")


def parse_expression(tokens, position=0):
    token = tokens[position]
    if token in UNARY_FUNCTIONS:
        if tokens[position + 1] != "(":
            raise ValueError(f"Expected '(' after unary function '{token}'")
        argument, next_position = parse_expression(tokens, position + 2)
        if tokens[next_position] != ")":
            raise ValueError(f"Expected ')' after unary function '{token}'")
        value = UNARY_FUNCTIONS[token](argument)
        position = next_position + 1
    elif token == "(":
        left_value, next_position = parse_expression(tokens, position + 1)
        if tokens[next_position] in BINARY_OPERATORS:
            operator_name = tokens[next_position]
            right_value, next_position = parse_expression(tokens, next_position + 1)
            if tokens[next_position] != ")":
                raise ValueError(f"Expected ')' to close binary expression with '{operator_name}'")
            value = apply_binary_operator(operator_name, left_value, right_value)
            position = next_position + 1
        else:
            if tokens[next_position] != ")":
                raise ValueError("Expected ')' to close grouped expression")
            value = left_value
            position = next_position + 1
    else:
        value, position = parse_atom(tokens, position)

    while position < len(tokens) and tokens[position] == "**":
        exponent, position = parse_atom(tokens, position + 1)
        value = value ** exponent
    return value, position


def tpsr_expr_to_sympy(expr_text):
    tokens = tokenize_expression(expr_text)
    expr, position = parse_expression(tokens, 0)
    if position != len(tokens):
        remaining = " ".join(tokens[position:])
        raise ValueError(f"Unparsed expression tail: {remaining}")
    return expr


def output_path_for(input_path):
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_simplified.csv")


def ensure_schema(fieldnames):
    required_columns = {"dataset", "status", "expr"}
    missing_columns = required_columns - set(fieldnames or [])
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")


def load_dataset_splits(dataset_name, datasets_dir, max_rows, max_input_points, dataset_cache):
    if dataset_name in dataset_cache:
        return dataset_cache[dataset_name]

    file_path = Path(datasets_dir) / dataset_name / f"{dataset_name}.tsv.gz"
    X, y, _ = read_file(str(file_path))
    y = np.expand_dims(y, -1)
    x_to_fit, x_to_predict, y_to_fit, y_to_predict = select_training_data(
        X,
        y,
        max_rows=max_rows,
        max_input_points=max_input_points,
    )
    dataset_cache[dataset_name] = (X.shape[0], x_to_fit, x_to_predict, y_to_fit, y_to_predict)
    return dataset_cache[dataset_name]


def has_non_finite_sympy(expr):
    return bool(expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo))


def should_fallback_to_original(simplified_expr, simplified_tree, y_pred, metrics):
    if simplified_tree is None:
        return True
    if has_non_finite_sympy(simplified_expr):
        return True
    if not np.isfinite(y_pred).all():
        return True
    if not np.isfinite(metrics["r2"][0]):
        return True
    if not np.isfinite(metrics["_complexity"][0]):
        return True
    return False


def simplify_row(row, env, datasets_dir, max_rows, max_input_points, dataset_cache):
    if (row["status"] or "").lower() not in SUCCESS_STATUSES or not row["expr"]:
        row["expr_simplified"] = ""
        return row

    total_rows, _, x_to_predict, _, y_to_predict = load_dataset_splits(
        row["dataset"],
        datasets_dir,
        max_rows=max_rows,
        max_input_points=max_input_points,
        dataset_cache=dataset_cache,
    )
    try:
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.alarm(ROW_TIMEOUT_SECONDS)
        original_expr = tpsr_expr_to_sympy(row["expr"])
        simplified_expr = env.simplifier.simplify_expr(original_expr)
        simplified_tree = env.simplifier.sympy_expr_to_tree(simplified_expr)
        y_pred = simplified_tree.val(np.copy(x_to_predict))
        if len(y_pred.shape) == 1:
            y_pred = np.expand_dims(y_pred, -1)
        metrics = compute_metrics(
            {
                "true": [y_to_predict],
                "predicted": [y_pred],
                "predicted_tree": [simplified_tree],
            },
            metrics="r2,_rmse,_complexity",
        )
        if should_fallback_to_original(simplified_expr, simplified_tree, y_pred, metrics):
            row["expr_simplified"] = row["expr"]
            return row
        row["expr_simplified"] = str(simplified_expr)
        row["r2"] = metrics["r2"][0]
        row["rmse"] = metrics["_rmse"][0]
        row["complexity"] = int(metrics["_complexity"][0])
    except Exception:
        row["expr_simplified"] = row["expr"]
        return row
    finally:
        signal.alarm(0)
    if "rows" in row:
        row["rows"] = total_rows
    if "noise_strength" in row:
        row["noise_strength"] = parse_noise_strength(row["_input_csv"])
    return row


def process_csv(input_path, env, datasets_dir, max_rows, max_input_points, dataset_cache):
    input_path = Path(input_path)
    output_path = output_path_for(input_path)
    with input_path.open(newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        ensure_schema(reader.fieldnames)
        fieldnames = list(reader.fieldnames)
        if "expr_simplified" not in fieldnames:
            expr_index = fieldnames.index("expr") + 1
            fieldnames.insert(expr_index, "expr_simplified")

        total_rows = sum(1 for _ in reader)
        input_handle.seek(0)
        reader = csv.DictReader(input_handle)
        with output_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(reader, start=1):
                start_time = time.time()
                row["_input_csv"] = str(input_path)
                simplified_row = simplify_row(
                    row,
                    env,
                    datasets_dir=datasets_dir,
                    max_rows=max_rows,
                    max_input_points=max_input_points,
                    dataset_cache=dataset_cache,
                )
                simplified_row.pop("_input_csv")
                writer.writerow({field: simplified_row.get(field, "") for field in fieldnames})
                output_handle.flush()
                row_seconds = time.time() - start_time
                if row_seconds > 10:
                    print(
                        f"[slow-row] {input_path.name} {index}/{total_rows} "
                        f"{row['dataset']} {row_seconds:.2f}s",
                        flush=True,
                    )
                if index % 25 == 0 or index == total_rows:
                    print(f"[{input_path.name}] {index}/{total_rows}", flush=True)

    print(f"Wrote simplified results to {output_path}", flush=True)


def main():
    args = parse_args()
    params = get_default_params()
    params.device = "cpu"
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    dataset_cache = {}

    for input_csv in args.input_csvs:
        process_csv(
            input_csv,
            env,
            datasets_dir=args.datasets_dir,
            max_rows=args.max_rows,
            max_input_points=args.max_input_points,
            dataset_cache=dataset_cache,
        )


if __name__ == "__main__":
    main()
