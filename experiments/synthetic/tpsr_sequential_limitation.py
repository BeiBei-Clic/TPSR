#!/usr/bin/env python3
"""
Minimal synthetic experiment for exposing TPSR's sequential-expansion limitation.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd
import sympy as sp
import torch
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.sklearn_wrapper import (
    SymbolicTransformerRegressor,
    get_top_k_features,
)
import symbolicregression.model.utils_wrapper as utils_wrapper
from tpsr import tpsr_fit


RESULT_COLUMNS = [
    "function_name",
    "true_expr",
    "target_operator",
    "seed",
    "n_points",
    "x_min",
    "x_max",
    "status",
    "expr",
    "r2",
    "rmse",
    "normalized_rmse",
    "complexity",
    "contains_target_operator",
    "is_polynomial",
    "is_taylor_like",
    "poly_degree",
    "seconds",
]


X_SYMBOL = sp.Symbol("x_0", real=True)
FUNCTION_SPECS = {
    "sin": {
        "true_expr": "sin(x)",
        "target_operator": "sin",
        "fn": np.sin,
    },
    "exp": {
        "true_expr": "exp(x)",
        "target_operator": "exp",
        "fn": np.exp,
    },
    "log1p": {
        "true_expr": "log(1 + x)",
        "target_operator": "log",
        "fn": np.log1p,
    },
}


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
    params.device = "cuda" if torch.cuda.is_available() else "cpu"
    return params


def parse_args():
    parser = argparse.ArgumentParser(description="TPSR synthetic sequential-limitation experiment")
    parser.add_argument(
        "--functions",
        type=str,
        default="all",
        help="Comma-separated function names from {sin,exp,log1p}, or 'all'",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=10,
        help="Number of random seeds to run per function",
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="Starting seed value",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=100,
        help="Number of sampled points per function",
    )
    parser.add_argument(
        "--x_range",
        type=float,
        nargs=2,
        default=(-0.5, 0.5),
        metavar=("X_MIN", "X_MAX"),
        help="Sampling range for x",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Additive Gaussian noise std on y",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="symbolicregression/saved_models/model.pt",
        help="Path to the TPSR pretrained model checkpoint",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="experiments/synthetic/results/tpsr_sequential_limitation.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. cuda, cuda:0, cpu",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Backward-compatible GPU id alias for --device cuda:N",
    )
    parser.add_argument(
        "--max_input_points",
        type=int,
        default=100,
        help="Maximum number of points observed by TPSR",
    )
    parser.add_argument(
        "--n_trees_to_refine",
        type=int,
        default=10,
        help="Number of candidate trees to refine",
    )
    parser.add_argument(
        "--taylor_rmse_ratio",
        type=float,
        default=0.05,
        help="Normalized RMSE threshold for tagging low-error Taylor-like compensation",
    )
    return parser.parse_args()


def parse_device(device_arg, gpu_arg):
    if device_arg and gpu_arg is not None:
        raise ValueError("Use either --device or --gpu, not both.")
    if device_arg:
        device = device_arg
    elif gpu_arg is not None:
        device = f"cuda:{gpu_arg}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        torch.cuda.set_device(index)
    return device


def parse_function_names(functions_arg):
    if functions_arg == "all":
        return list(FUNCTION_SPECS.keys())
    names = [name.strip() for name in functions_arg.split(",") if name.strip()]
    invalid = [name for name in names if name not in FUNCTION_SPECS]
    if invalid:
        raise ValueError(f"Unknown function names: {invalid}")
    return names


def validate_args(args):
    selected = parse_function_names(args.functions)
    if args.num_seeds <= 0:
        raise ValueError("--num_seeds must be positive.")
    if args.n_points <= 0:
        raise ValueError("--n_points must be positive.")
    if args.max_input_points <= 0:
        raise ValueError("--max_input_points must be positive.")
    if args.noise_std < 0:
        raise ValueError("--noise_std must be non-negative.")
    if args.taylor_rmse_ratio < 0:
        raise ValueError("--taylor_rmse_ratio must be non-negative.")
    x_min, x_max = args.x_range
    if x_min >= x_max:
        raise ValueError("--x_range must satisfy X_MIN < X_MAX.")
    if "log1p" in selected and x_min <= -1:
        raise ValueError("log1p requires x > -1 across the whole sampling range.")


def load_model(params, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    env = build_env(params)
    env.rng = np.random.RandomState(0)

    model_wrapper = torch.load(model_path, map_location=params.device, weights_only=False)
    model_wrapper.to(params.device)
    model_wrapper.eval()
    return env, model_wrapper


def sample_function_data(function_name, seed, n_points, x_range, noise_std):
    spec = FUNCTION_SPECS[function_name]
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], size=(n_points, 1))
    y = spec["fn"](x[:, 0]).reshape(-1, 1)
    if noise_std > 0:
        y = y + rng.normal(loc=0.0, scale=noise_std, size=y.shape)
    return x, y


def run_tpsr_on_points(X, y, params, env, model_wrapper):
    X_list = [X[: params.max_input_points]]
    Y_list = [y[: params.max_input_points]]

    top_k_features = get_top_k_features(X_list[0], Y_list[0], k=params.max_input_dimension)
    X_list[0] = X_list[0][:, top_k_features]

    scaler = utils_wrapper.StandardScaler() if params.rescale else None
    scale_params = {}
    if scaler is not None:
        scaled_X = []
        for i, x_values in enumerate(X_list):
            scaled_X.append(scaler.fit_transform(x_values))
            scale_params[i] = scaler.get_params()
    else:
        scaled_X = X_list

    tpsr_start = time.time()
    sequence, _, _ = tpsr_fit(
        scaled_X,
        Y_list,
        params,
        env,
        bag_number=1,
        rescale=params.rescale,
    )
    total_seconds = time.time() - tpsr_start

    generated_tree = [
        tree
        for tree in [env.idx_to_infix(sequence[1:], is_float=False, str_array=False)]
        if tree is not None
    ]
    if not generated_tree:
        raise ValueError("No valid equation generated.")

    dstr = SymbolicTransformerRegressor(
        model=model_wrapper,
        max_input_points=params.max_input_points,
        n_trees_to_refine=params.n_trees_to_refine,
        max_number_bags=params.max_number_bags,
        rescale=params.rescale,
    )
    dstr.top_k_features = [top_k_features]
    dstr.start_fit = time.time()
    dstr.tree = {}
    refined_candidates = dstr.refine(scaled_X[0], Y_list[0], generated_tree, verbose=False)

    if scaler is not None and refined_candidates:
        refined_candidates[0]["predicted_tree"] = scaler.rescale_function(
            env,
            refined_candidates[0]["predicted_tree"],
            *scale_params[0],
        )
    dstr.tree[0] = refined_candidates

    best_gen = dstr.retrieve_tree(refinement_type="BFGS", with_infos=True)
    predicted_tree = best_gen["predicted_tree"]
    if predicted_tree is None:
        raise ValueError("Refinement failed.")

    x_for_predict = X[:, top_k_features]
    y_pred = dstr.predict(x_for_predict, refinement_type="BFGS")
    if y_pred is None:
        raise ValueError("Prediction failed.")

    y_true = y[:, 0] if len(y.shape) == 2 else y
    y_pred_flat = y_pred[:, 0] if len(y_pred.shape) == 2 else y_pred
    metrics = {
        "r2": [float(r2_score(y_true, y_pred_flat))],
        "_rmse": [float(np.sqrt(mean_squared_error(y_true, y_pred_flat)))],
        "_complexity": [int(len(predicted_tree.prefix().split(",")))],
    }
    return predicted_tree, metrics, total_seconds


def expr_has_operator(expr, operator_name):
    target_func = {
        "sin": sp.sin,
        "exp": sp.exp,
        "log": sp.log,
    }[operator_name]
    for node in sp.preorder_traversal(expr):
        if getattr(node, "func", None) == target_func:
            return True
    return False


def polynomial_degree(expr):
    if len(expr.free_symbols) > 1:
        return "", False
    symbol = next(iter(expr.free_symbols)) if expr.free_symbols else X_SYMBOL
    if not expr.is_polynomial(symbol):
        return "", False
    poly = sp.Poly(sp.expand(expr), symbol)
    return int(poly.total_degree()), True


def analyze_expression(env, predicted_tree, target_operator, rmse, y_true, taylor_rmse_ratio):
    expr = sp.simplify(env.simplifier.tree_to_sympy_expr(predicted_tree))
    expr_text = sp.sstr(expr)
    contains_target_operator = expr_has_operator(expr, target_operator)
    poly_degree, is_polynomial = polynomial_degree(expr)
    y_scale = float(np.std(y_true))
    normalized_rmse = float(rmse / max(y_scale, 1e-12))
    is_taylor_like = (
        not contains_target_operator
        and is_polynomial
        and normalized_rmse <= taylor_rmse_ratio
    )
    return {
        "expr": expr_text,
        "contains_target_operator": contains_target_operator,
        "is_polynomial": is_polynomial,
        "is_taylor_like": is_taylor_like,
        "poly_degree": poly_degree,
        "normalized_rmse": normalized_rmse,
    }


def make_row(function_name, spec, seed, args, env, predicted_tree, metrics, total_seconds, y_true):
    rmse = float(metrics["_rmse"][0])
    analysis = analyze_expression(
        env,
        predicted_tree,
        spec["target_operator"],
        rmse,
        y_true,
        args.taylor_rmse_ratio,
    )
    return {
        "function_name": function_name,
        "true_expr": spec["true_expr"],
        "target_operator": spec["target_operator"],
        "seed": seed,
        "n_points": args.n_points,
        "x_min": args.x_range[0],
        "x_max": args.x_range[1],
        "status": "success",
        "expr": analysis["expr"],
        "r2": float(metrics["r2"][0]),
        "rmse": rmse,
        "normalized_rmse": analysis["normalized_rmse"],
        "complexity": int(metrics["_complexity"][0]),
        "contains_target_operator": analysis["contains_target_operator"],
        "is_polynomial": analysis["is_polynomial"],
        "is_taylor_like": analysis["is_taylor_like"],
        "poly_degree": analysis["poly_degree"],
        "seconds": total_seconds,
    }


def summarize_results(results):
    if not results:
        return

    summary_df = pd.DataFrame(results)
    print(f"Completed runs: {len(summary_df)}")
    for function_name in summary_df["function_name"].unique():
        subset = summary_df[summary_df["function_name"] == function_name]
        true_struct = subset[subset["contains_target_operator"]]
        compensated = subset[subset["is_taylor_like"]]
        print(
            f"{function_name}: "
            f"target_op_rate={len(true_struct) / len(subset):.3f}, "
            f"taylor_like_rate={len(compensated) / len(subset):.3f}, "
            f"mean_complexity={subset['complexity'].mean():.2f}"
        )
        if len(true_struct) > 0:
            print(f"{function_name}: true_structure_complexity={true_struct['complexity'].mean():.2f}")
        if len(compensated) > 0:
            print(f"{function_name}: taylor_like_complexity={compensated['complexity'].mean():.2f}")



def write_result_row(writer, output_handle, row):
    ordered_row = {column: row.get(column, "") for column in RESULT_COLUMNS}
    writer.writerow(ordered_row)
    output_handle.flush()


def main():
    args = parse_args()
    validate_args(args)

    function_names = parse_function_names(args.functions)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    device = parse_device(args.device, args.gpu)

    params = get_default_params()
    params.device = device
    params.max_input_points = args.max_input_points
    params.n_trees_to_refine = args.n_trees_to_refine

    env, model_wrapper = load_model(params, args.model_path)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    with open(args.output_csv, "w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        output_handle.flush()

        for function_name in function_names:
            spec = FUNCTION_SPECS[function_name]
            for seed in seeds:
                X, y = sample_function_data(
                    function_name,
                    seed,
                    args.n_points,
                    args.x_range,
                    args.noise_std,
                )
                predicted_tree, metrics, total_seconds = run_tpsr_on_points(
                    X,
                    y,
                    params,
                    env,
                    model_wrapper,
                )
                row = make_row(function_name, spec, seed, args, env, predicted_tree, metrics, total_seconds, y)
                results.append(row)
                write_result_row(writer, output_handle, row)

    summarize_results(results)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
