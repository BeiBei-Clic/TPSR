#!/usr/bin/env python3
"""
TPSR PMLB inference with batch execution, resumable CSV output, and target-noise control.
"""

import argparse
import csv
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.metrics import compute_metrics
from symbolicregression.model.sklearn_wrapper import (
    SymbolicTransformerRegressor,
    get_top_k_features,
)
import symbolicregression.model.utils_wrapper as utils_wrapper
from tpsr import tpsr_fit


RESULT_COLUMNS = [
    "dataset",
    "status",
    "n_features",
    "refinement_type",
    "r2",
    "rmse",
    "complexity",
    "seconds",
    "error",
    "noise_strength",
    "expr",
]


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
    params.device = "cuda"

    return params


def parse_args():
    parser = argparse.ArgumentParser(description="TPSR PMLB Inference")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 1027_ESL) or 'all' for all datasets",
    )
    parser.add_argument(
        "--datasets_dir",
        "--data_path",
        dest="datasets_dir",
        type=str,
        default="pmlb/datasets",
        help="Path to the PMLB datasets directory",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default=None,
        choices=["feynman", "strogatz", "black-box"],
        help="PMLB data group filter (only used when --dataset all)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="symbolicregression/saved_models/model.pt",
        help="Path to the TPSR pretrained model checkpoint",
    )
    parser.add_argument(
        "--output_csv",
        "--output",
        dest="output_csv",
        type=str,
        default=None,
        help="Output CSV path; default includes the noise strength in the file name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. cuda, cuda:0, cuda:1",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Backward-compatible GPU id alias for --device cuda:N",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Maximum number of rows to load from each dataset before splitting",
    )
    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="Maximum number of training points used by TPSR",
    )
    parser.add_argument(
        "--n_trees_to_refine",
        type=int,
        default=10,
        help="Number of candidate trees to refine",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=None,
        help="Only process the first N matched datasets",
    )
    parser.add_argument(
        "--noise_strength",
        type=float,
        default=0.0,
        help="Multiplicative Gaussian target-noise strength eps in y * (1 + N(0, eps)); 0 keeps the original behavior",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=0,
        help="Base random seed for target-noise generation",
    )
    return parser.parse_args()


def read_file(filename, label="target", sep=None):
    compression = "gzip" if filename.endswith("gz") else None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression, engine="python")
    feature_names = np.array([x for x in input_data.columns.values if x != label])
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]
    return X, y, feature_names


def parse_device(device_arg, gpu_arg):
    if device_arg and gpu_arg is not None:
        raise ValueError("Use either --device or --gpu, not both.")
    if device_arg:
        device = device_arg
    elif gpu_arg is not None:
        device = f"cuda:{gpu_arg}"
    else:
        device = "cuda"

    if not device.startswith("cuda"):
        raise ValueError("TPSR PMLB inference only supports CUDA devices. Use --device cuda or --device cuda:N.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")
    if device == "cuda":
        return device
    if not device.startswith("cuda:"):
        raise ValueError(f"Unsupported device format: {device}")
    index_text = device.split(":", 1)[1]
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"Invalid CUDA device index in '{device}'") from exc
    if index < 0:
        raise ValueError(f"Invalid CUDA device index in '{device}'")
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are available."
        )
    torch.cuda.set_device(index)
    return device


def format_noise_strength(value):
    return np.format_float_positional(float(value), trim="-")


def default_output_csv(noise_strength):
    noise_text = format_noise_strength(noise_strength)
    return os.path.join(
        "experiments",
        "pmlb",
        "results",
        f"pmlb_batch_inference_noise_{noise_text}.csv",
    )


def metadata_value(metadata_path, key):
    if not os.path.exists(metadata_path):
        return None
    prefix = f"{key}:"
    with open(metadata_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped.split(":", 1)[1].strip()
    return None


def classify_data_group(dataset_name):
    if dataset_name.startswith("feynman_"):
        return "feynman"
    if dataset_name.startswith("strogatz_"):
        return "strogatz"
    return "black-box"


def list_regression_datasets(datasets_dir, data_type=None):
    datasets = []
    for dataset_name in sorted(os.listdir(datasets_dir)):
        dataset_dir = os.path.join(datasets_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        metadata_path = os.path.join(dataset_dir, "metadata.yaml")
        if metadata_value(metadata_path, "task") != "regression":
            continue
        if data_type is not None and classify_data_group(dataset_name) != data_type:
            continue
        datasets.append(dataset_name)
    return datasets


def make_result(dataset_name, noise_strength, **overrides):
    row = {
        "dataset": dataset_name,
        "status": "failed",
        "n_features": np.nan,
        "refinement_type": "",
        "r2": np.nan,
        "rmse": np.nan,
        "complexity": np.nan,
        "seconds": np.nan,
        "error": "",
        "noise_strength": noise_strength,
        "expr": "",
    }
    row.update(overrides)
    return row


def validate_existing_output(output_csv):
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return
    with open(output_csv, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header != RESULT_COLUMNS:
        raise ValueError(
            f"Existing CSV header does not match expected schema for {output_csv}. "
            "Please use a new output path or remove the old file first."
        )


def load_completed_datasets(output_csv):
    if not os.path.exists(output_csv):
        return set()
    completed = set()
    with open(output_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "dataset" not in (reader.fieldnames or []):
            return set()
        for row in reader:
            dataset_name = row.get("dataset")
            if dataset_name:
                completed.add(dataset_name)
    return completed


def noise_rng_for_dataset(noise_seed, dataset_name):
    digest = hashlib.md5(dataset_name.encode("utf-8")).hexdigest()
    dataset_offset = int(digest[:8], 16)
    seed = (int(noise_seed) + dataset_offset) % (2 ** 32)
    return np.random.RandomState(seed)


def maybe_add_target_noise(y_to_fit, noise_strength, rng):
    if noise_strength < 0:
        raise ValueError("--noise_strength must be non-negative.")
    if noise_strength == 0:
        return y_to_fit
    noise = rng.normal(loc=0.0, scale=noise_strength, size=y_to_fit.shape)
    return y_to_fit * (1.0 + noise)


def load_model(params, model_path):
    print(f"Loading model from {model_path}...")
    print(f"Device: {params.device}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    env = build_env(params)
    env.rng = np.random.RandomState(0)

    model_wrapper = torch.load(model_path, map_location=params.device, weights_only=False)
    model_wrapper.to(params.device)
    model_wrapper.eval()

    print("Model loaded successfully")
    return env, model_wrapper


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


def run_tpsr_inference(dataset_name, datasets_dir, params, env, model_wrapper, noise_strength, noise_seed, max_rows):
    print(f"\n{'=' * 60}")
    print(f"Running TPSR inference on: {dataset_name}")
    print(f"{'=' * 60}")

    dataset_dir = os.path.join(datasets_dir, dataset_name)
    metadata_path = os.path.join(dataset_dir, "metadata.yaml")
    task_type = metadata_value(metadata_path, "task")
    if task_type and task_type != "regression":
        return make_result(
            dataset_name,
            noise_strength,
            status="skipped",
            error=f"Unsupported task type: {task_type}",
        )

    file_path = os.path.join(dataset_dir, f"{dataset_name}.tsv.gz")
    if not os.path.exists(file_path):
        return make_result(
            dataset_name,
            noise_strength,
            status="failed",
            error=f"Dataset file not found: {file_path}",
        )

    start_time = time.time()
    try:
        X, y, feature_names = read_file(file_path)
        y = np.expand_dims(y, -1)
        n_features = int(len(feature_names))

        print(f"Dataset shape: X={X.shape}, y={y.shape}")
        print(f"Number of features: {n_features}")

        if X.shape[1] > params.max_input_dimension:
            return make_result(
                dataset_name,
                noise_strength,
                status="skipped",
                n_features=n_features,
                error=(
                    f"Input dimension {X.shape[1]} exceeds "
                    f"max_input_dimension={params.max_input_dimension}"
                ),
            )

        x_to_fit, x_to_predict, y_to_fit, y_to_predict = select_training_data(
            X, y, max_rows=max_rows, max_input_points=params.max_input_points
        )
        rng = noise_rng_for_dataset(noise_seed, dataset_name)
        y_to_fit = maybe_add_target_noise(y_to_fit, noise_strength, rng)

        X_list = [x_to_fit]
        Y_list = [y_to_fit]

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
        sequence, sample_seconds, _ = tpsr_fit(
            scaled_X,
            Y_list,
            params,
            env,
            bag_number=1,
            rescale=params.rescale,
        )
        total_seconds = time.time() - tpsr_start

        print(f"TPSR sampling time: {sample_seconds:.2f}s")
        print(f"Total time: {total_seconds:.2f}s")

        generated_tree = [
            tree
            for tree in [env.idx_to_infix(sequence[1:], is_float=False, str_array=False)]
            if tree is not None
        ]
        if not generated_tree:
            return make_result(
                dataset_name,
                noise_strength,
                status="failed",
                n_features=n_features,
                seconds=total_seconds,
                error="No valid equation generated",
            )

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
            return make_result(
                dataset_name,
                noise_strength,
                status="failed",
                n_features=n_features,
                seconds=total_seconds,
                error="Refinement failed",
            )

        expr = predicted_tree.infix()
        complexity = len(predicted_tree.prefix().split(","))
        print(f"Expression: {expr}")
        print(f"Complexity: {complexity}")

        x_to_predict_selected = x_to_predict[:, top_k_features]
        y_pred = dstr.predict(x_to_predict_selected, refinement_type="BFGS")
        if y_pred is None:
            return make_result(
                dataset_name,
                noise_strength,
                status="failed",
                n_features=n_features,
                refinement_type="BFGS",
                complexity=complexity,
                seconds=total_seconds,
                expr=expr,
                error="Prediction failed",
            )

        metrics = compute_metrics(
            {
                "true": [y_to_predict],
                "predicted": [y_pred],
                "predicted_tree": [predicted_tree],
            },
            metrics="r2,_rmse,_complexity",
        )
        r2 = metrics["r2"][0]
        rmse_val = metrics["_rmse"][0]

        print(f"R2: {r2:.4f}")
        print(f"RMSE: {rmse_val:.4f}")

        return make_result(
            dataset_name,
            noise_strength,
            status="success",
            n_features=n_features,
            refinement_type="BFGS",
            r2=r2,
            rmse=rmse_val,
            complexity=complexity,
            seconds=total_seconds,
            expr=expr,
            error="",
        )
    except Exception as exc:
        return make_result(
            dataset_name,
            noise_strength,
            status="failed",
            seconds=time.time() - start_time,
            error=str(exc),
        )


def write_result_row(writer, output_handle, row):
    ordered_row = {column: row.get(column, "") for column in RESULT_COLUMNS}
    writer.writerow(ordered_row)
    output_handle.flush()


def summarize_results(results_list):
    if not results_list:
        return
    summary_df = pd.DataFrame(results_list)
    numeric_columns = ["r2", "rmse", "complexity", "seconds"]
    available_columns = [column for column in numeric_columns if column in summary_df.columns]
    if available_columns:
        print("\nSummary Statistics:")
        print(summary_df[available_columns].describe())

    successful = summary_df[summary_df["status"] == "success"]
    print(f"\nCompleted rows: {len(summary_df)}")
    print(f"Successful rows: {len(successful)}")
    if not successful.empty:
        print(f"Mean R2: {successful['r2'].mean():.4f}")
        print(f"Std R2: {successful['r2'].std():.4f}")


def main():
    args = parse_args()

    if args.noise_strength < 0:
        raise ValueError("--noise_strength must be non-negative.")
    if args.dataset_limit is not None and args.dataset_limit <= 0:
        raise ValueError("--dataset_limit must be positive when provided.")

    device = parse_device(args.device, args.gpu)
    output_csv = args.output_csv or default_output_csv(args.noise_strength)

    params = get_default_params()
    params.device = device
    params.max_input_points = args.max_input_points
    params.n_trees_to_refine = args.n_trees_to_refine

    env, model_wrapper = load_model(params, args.model_path)

    if args.dataset == "all":
        datasets = list_regression_datasets(args.datasets_dir, data_type=args.data_type)
        if args.dataset_limit is not None:
            datasets = datasets[:args.dataset_limit]
    else:
        datasets = [args.dataset]

    print(f"\nProcessing {len(datasets)} dataset(s)...")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    validate_existing_output(output_csv)
    completed_datasets = load_completed_datasets(output_csv)
    if completed_datasets:
        print(f"Found {len(completed_datasets)} completed dataset(s) in {output_csv}")

    results_list = []
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0

    with open(output_csv, "a", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
            output_handle.flush()

        for dataset_name in datasets:
            if dataset_name in completed_datasets:
                print(f"Skipping {dataset_name}: already exists in {output_csv}")
                continue

            result = run_tpsr_inference(
                dataset_name,
                args.datasets_dir,
                params,
                env,
                model_wrapper,
                noise_strength=args.noise_strength,
                noise_seed=args.noise_seed,
                max_rows=args.max_rows,
            )
            results_list.append(result)
            completed_datasets.add(dataset_name)
            write_result_row(writer, output_handle, result)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_csv}")
    print(f"{'=' * 60}")
    summarize_results(results_list)


if __name__ == "__main__":
    main()
