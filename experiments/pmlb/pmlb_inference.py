#!/usr/bin/env python3
"""
TPSR PMLB Inference Script

This script runs TPSR (Transformer-based Planning for Symbolic Regression)
inference on PMLB datasets.

Usage:
    python pmlb_inference.py --dataset 1027_ESL
    python pmlb_inference.py --dataset 1027_ESL --gpu 0
    python pmlb_inference.py --dataset all --data_type feynman
"""

import argparse
import os
import sys
import time
import copy
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from sklearn.model_selection import train_test_split

from symbolicregression.envs import build_env
from symbolicregression.e2e_model import Transformer
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor, get_top_k_features
from symbolicregression.metrics import compute_metrics
import symbolicregression.model.utils_wrapper as utils_wrapper
from tpsr import tpsr_fit
from parsers import get_parser


def get_default_params():
    """Get default parameters using the existing parser"""
    parser = get_parser()
    params, unknown = parser.parse_known_args([])

    # Override with TPSR recommended parameters
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

    # Device
    params.device = "cuda" if torch.cuda.is_available() else "cpu"

    return params


def read_file(filename, label="target", sep=None):
    """Read PMLB dataset file"""
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


def load_model(params):
    """Load the pre-trained E2E model and build environment"""
    model_path = "symbolicregression/saved_models/model.pt"
    print(f"Loading model from {model_path}...")
    print(f"Device: {params.device}")

    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Build environment
    env = build_env(params)
    env.rng = np.random.RandomState(0)

    # Load model checkpoint (it's a ModelWrapper object)
    model_wrapper = torch.load(model_path, map_location=params.device, weights_only=False)

    # Move model components to device and set to eval mode
    model_wrapper.to(params.device)
    model_wrapper.eval()

    print("Model loaded successfully")
    return env, model_wrapper


def load_completed_datasets(output_path):
    """Load dataset names that already exist in the output CSV."""
    if not os.path.exists(output_path):
        return set()

    completed = set()
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        if "dataset" not in (reader.fieldnames or []):
            return set()
        for row in reader:
            dataset_name = row.get("dataset")
            if dataset_name:
                completed.add(dataset_name)
    return completed


def run_tpsr_inference(dataset_name, data_path, params, env, model_wrapper):
    """Run TPSR inference on a single dataset"""
    print(f"\n{'='*60}")
    print(f"Running TPSR inference on: {dataset_name}")
    print(f"{'='*60}")

    # Read dataset
    file_path = os.path.join(data_path, dataset_name, f"{dataset_name}.tsv.gz")
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found: {file_path}")
        return None

    X, y, feature_names = read_file(file_path)
    y = np.expand_dims(y, -1)

    # Limit to max_input_points
    if len(X) > params.max_input_points:
        X = X[:params.max_input_points]
        y = y[:params.max_input_points]

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Number of features: {len(feature_names)}")

    if X.shape[1] > params.max_input_dimension:
        print(
            f"Skipping {dataset_name}: input dimension {X.shape[1]} "
            f"exceeds max_input_dimension={params.max_input_dimension}"
        )
        return None

    # Split data
    x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=29910
    )

    # Prepare data for TPSR
    if not isinstance(x_to_fit, list):
        X_list = [x_to_fit]
        Y_list = [y_to_fit]

    # Feature selection
    top_k_features = get_top_k_features(X_list[0], Y_list[0], k=params.max_input_dimension)
    X_list[0] = X_list[0][:, top_k_features]

    # Scale data
    scaler = utils_wrapper.StandardScaler() if params.rescale else None
    scale_params = {}
    if scaler is not None:
        scaled_X = []
        for i, x in enumerate(X_list):
            scaled_X.append(scaler.fit_transform(x))
            scale_params[i] = scaler.get_params()
    else:
        scaled_X = X_list

    # Run TPSR inference
    start_time = time.time()
    s, time_elapsed, sample_times = tpsr_fit(scaled_X, Y_list, params, env, bag_number=1, rescale=params.rescale)
    total_time = time.time() - start_time

    print(f"TPSR sampling time: {time_elapsed:.2f}s")
    print(f"Total time: {total_time:.2f}s")

    # Convert to tree
    generated_tree = list(filter(lambda x: x is not None,
        [env.idx_to_infix(s[1:], is_float=False, str_array=False)]))

    if generated_tree == []:
        print("Warning: No valid equation generated")
        return {
            "dataset": dataset_name,
            "r2": np.nan,
            "rmse": np.nan,
            "time": total_time,
            "complexity": np.nan,
            "expression": None
        }

    # Create model wrapper for refinement
    dstr = SymbolicTransformerRegressor(
        model=model_wrapper,
        max_input_points=params.max_input_points,
        n_trees_to_refine=params.n_trees_to_refine,
        max_number_bags=params.max_number_bags,
        rescale=params.rescale,
    )
    dstr.top_k_features = [top_k_features]

    # Refine the equation
    dstr.start_fit = time.time()
    dstr.tree = {}
    refined_candidate = dstr.refine(scaled_X[0], Y_list[0], generated_tree, verbose=False)

    if scaler is not None:
        refined_candidate[0]["predicted_tree"] = scaler.rescale_function(
            env, refined_candidate[0]["predicted_tree"], *scale_params[0]
        )
    dstr.tree[0] = refined_candidate

    # Get best refined tree
    best_gen = dstr.retrieve_tree(refinement_type="BFGS", with_infos=True)
    predicted_tree = best_gen["predicted_tree"]

    if predicted_tree is None:
        print("Warning: Refinement failed")
        return {
            "dataset": dataset_name,
            "r2": np.nan,
            "rmse": np.nan,
            "time": total_time,
            "complexity": np.nan,
            "expression": None
        }

    # Get expression
    expression = predicted_tree.infix()
    print(f"Expression: {expression}")

    # Get complexity
    complexity = len(predicted_tree.prefix().split(","))
    print(f"Complexity: {complexity}")

    # Predict on test set
    x_to_predict_selected = x_to_predict[:, top_k_features]
    y_pred = dstr.predict(x_to_predict_selected, refinement_type="BFGS")

    if y_pred is None:
        print("Warning: Prediction failed")
        return {
            "dataset": dataset_name,
            "r2": np.nan,
            "rmse": np.nan,
            "time": total_time,
            "complexity": complexity,
            "expression": expression
        }

    # Compute metrics
    results = compute_metrics(
        {
            "true": [y_to_predict],
            "predicted": [y_pred],
            "predicted_tree": [predicted_tree],
        },
        metrics="r2,_rmse,_complexity"
    )

    r2 = results["r2"][0]
    rmse_val = results["_rmse"][0]

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse_val:.4f}")

    return {
        "dataset": dataset_name,
        "r2": r2,
        "rmse": rmse_val,
        "time": total_time,
        "complexity": complexity,
        "expression": expression
    }


def main():
    parser = argparse.ArgumentParser(description="TPSR PMLB Inference")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., 1027_ESL) or 'all' for all datasets")
    parser.add_argument("--data_path", type=str, default="pmlb/datasets",
                        help="Path to PMLB datasets directory")
    parser.add_argument("--data_type", type=str, default=None,
                        choices=["feynman", "strogatz", "black-box"],
                        help="PMLB data type (only used when --dataset all)")
    parser.add_argument("--output", type=str, default="experiments/pmlb/results/pmlb_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID (e.g., 0, 1)")
    parser.add_argument("--max_input_points", type=int, default=200,
                        help="Maximum number of input points")

    args = parser.parse_args()

    # Set GPU
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Create params with defaults
    params = get_default_params()
    params.max_input_points = args.max_input_points
    if args.gpu is not None and torch.cuda.is_available():
        params.device = f"cuda:{args.gpu}"

    # Load environment and model
    env, model_wrapper = load_model(params)

    # Determine datasets to process
    if args.dataset == "all":
        # Get all regression datasets
        all_datasets = sorted([d for d in os.listdir(args.data_path)
                               if os.path.isdir(os.path.join(args.data_path, d))])

        # Filter by data type if specified
        if args.data_type:
            # Simple heuristic: Feynman datasets start with specific patterns
            if args.data_type == "feynman":
                # Feynman datasets in PMLB have specific naming
                all_datasets = [d for d in all_datasets if d.startswith("feynman")]
            elif args.data_type == "strogatz":
                all_datasets = [d for d in all_datasets if "strogatz" in d.lower()]
            elif args.data_type == "black-box":
                all_datasets = [d for d in all_datasets if not d.startswith("feynman") and "strogatz" not in d.lower()]

        datasets = all_datasets
    else:
        datasets = [args.dataset]

    print(f"\nProcessing {len(datasets)} dataset(s)...")

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    completed_datasets = load_completed_datasets(args.output)
    if completed_datasets:
        print(f"Found {len(completed_datasets)} completed dataset(s) in {args.output}")

    # Process each dataset
    results_list = []
    first_write = not os.path.exists(args.output) or os.path.getsize(args.output) == 0

    for dataset_name in datasets:
        if dataset_name in completed_datasets:
            print(f"Skipping {dataset_name}: already exists in {args.output}")
            continue

        result = run_tpsr_inference(dataset_name, args.data_path, params, env, model_wrapper)

        if result:
            results_list.append(result)
            completed_datasets.add(dataset_name)

            # Append to CSV
            result_df = pd.DataFrame([result])
            if first_write:
                result_df.to_csv(args.output, index=False)
                first_write = False
            else:
                result_df.to_csv(args.output, mode='a', header=False, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")

    # Print summary statistics
    if results_list:
        summary_df = pd.DataFrame(results_list)
        print("\nSummary Statistics:")
        print(summary_df.describe())

        # Count valid results
        valid_r2 = summary_df["r2"].notna().sum()
        print(f"\nValid results (R2): {valid_r2}/{len(results_list)}")
        if valid_r2 > 0:
            print(f"Mean R2: {summary_df['r2'].mean():.4f}")
            print(f"Std R2: {summary_df['r2'].std():.4f}")


if __name__ == "__main__":
    main()
