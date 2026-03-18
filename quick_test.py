#!/usr/bin/env python3
"""
Quick test script for TPSR E2E demo verification
Tests basic functionality with minimal parameters
"""
import os
os.environ['PYTHONPATH'] = '/home/xyh/Symbolic_Regression/E2E/TPSR/nesymres/src'

import sys
sys.path.insert(0, '/home/xyh/Symbolic_Regression/E2E/TPSR/nesymres/src')

import time
import torch
import numpy as np
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer
from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine
from parsers import get_parser

def compute_nmse(y_gt, y_pred):
    eps = 1e-9
    return np.sqrt(np.mean((y_gt - y_pred)**2) / (np.mean((y_gt)**2) + eps))

def main():
    parser = get_parser()
    params = parser.parse_args([
        '--backbone_model', 'e2e',
        '--no_seq_cache', 'True',
        '--no_prefix_cache', 'True'
    ])

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {params.device}")

    # Build environment and model
    equation_env = build_env(params)
    modules = build_modules(equation_env, params)
    trainer = Trainer(modules, equation_env, params)

    # Create simple test data
    x0 = np.linspace(-2, 2, 100)
    y = (x0**2) * np.sin(5*x0) + np.exp(-0.5*x0)
    data = np.concatenate((x0.reshape(-1,1), y.reshape(-1,1)), axis=1)

    samples = {
        'x_to_fit': [data[:,:1]],
        'y_to_fit': [data[:,1].reshape(-1,1)],
        'x_to_pred': [data[:,:1]],
        'y_to_pred': [data[:,1].reshape(-1,1)]
    }

    print("Loading E2E model...")
    model = Transformer(params=params, env=equation_env, samples=samples)
    model.to(params.device)

    print("Running prediction...")
    start = time.time()

    # Generate sequence
    from symbolicregression.e2e_model import respond_to_batch
    generations_ref, gen_len_ref = respond_to_batch(model, max_target_length=200, top_p=1.0, sample_temperature=None)
    sequence_ref = generations_ref[0][:gen_len_ref-1].tolist()

    # Predict
    y_pred, pred_str, pred_tree = pred_for_sample_no_refine(
        model, equation_env, sequence_ref, samples['x_to_fit']
    )

    elapsed = time.time() - start

    # Calculate metrics
    y_gt = samples['y_to_fit'][0].reshape(-1)
    nmse = compute_nmse(y_gt, y_pred)

    # Print results
    print("\n" + "="*50)
    print("TPSR E2E Demo - Quick Test Results")
    print("="*50)
    print(f"Test equation: y = x^2 * sin(5*x) + exp(-0.5*x)")
    print(f"Number of points: {len(x0)}")
    print(f"Predicted equation: {pred_str}")
    print(f"NMSE: {nmse:.6f}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print("="*50)

    if nmse < 0.1:
        print("✓ Test PASSED - Good prediction quality!")
        return 0
    else:
        print("⚠ Test WARNING - High NMSE value")
        return 1

if __name__ == '__main__':
    sys.exit(main())
