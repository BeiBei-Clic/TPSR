# Deep Symbolic Regression with Transformers & Lookahead Planning

Official Implementation of **[Transformer-based Planning for Symbolic Regression](https://arxiv.org/abs/2303.06833)** (NeurIPS 2023). 

[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html) | [SRBench Results](https://github.com/deep-symbolic-mathematics/TPSR/tree/main/srbench_results) | [Code](https://github.com/deep-symbolic-mathematics/TPSR/)


## Overview
In this paper, we introduce **TPSR**, a novel transformer-based planning framework for symbolic regression by leveraging priors of large-scale pretrained models and incorporating lookahead planning. TPSR incorporates Monte Carlo Tree Search (MCTS) into the transformer decoding process of symbolic regression models. Unlike conventional decoding strategies, TPSR enables the integration of non-differentiable feedback, such as fitting accuracy and complexity, as external sources of knowledge into the transformer-based equation generation process.

<p align="center">
<img src="./images/Media13_Final.gif" width="100%" /> 
 <br>
<b>TPSR uncovering the governing symbolic mathematics of data, providing enhanced extrapolation capabilities.</b>
</p>


## Preperation: Data and Pre-trained Backbone Models 
<!-- ### Download Manually -->
1. Download Pre-trained Models:
    * **End-to-End (E2E) SR Transformer model** is available [here](https://dl.fbaipublicfiles.com/symbolicregression/model1.pt).
    * **NeSymReS model** is available [here](https://drive.google.com/file/d/1cNZq3dLnSUKEm-ujDl2mb2cCCorv7kOC/).
      
After downloading, extract both models to this directory. They should be located under the `symbolicregression/weights/` and `nesymres/weights/` sub-folders, respectively.

2. Download Benchmark Datasets:
    * **Feynman** equations are [here](https://space.mit.edu/home/tegmark/aifeynman.html)
    * **PMLB** datasets are also [here](https://github.com/EpistasisLab/pmlb/tree/master/datasets). Data points of PMLB datasets are used in the [SRBench (A Living Benchmark for Symbolic Regression)](https://github.com/cavalab/srbench), containing three data groups: **Feynman**, **Strogatz**, and **Black-box**.
      
Extract the datasets to this directory, Feynman datasets should be in `datasets/feynman/`, and PMLB datasets should be in `datasets/pmlb/`. 


## Installation

To run the code with deafult [E2E](https://arxiv.org/pdf/2204.10532.pdf) backbone model, create a conda environment and install the dependencies by running the following command.

```
conda create --name tpsr
conda activate tpsr
pip install -r requirements.txt
```

If you're interested to run experiments with [NeSymReS](https://arxiv.org/pdf/2106.06427.pdf) backbone, install its additional dependencies from [here](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales). You can follow these steps:

```
conda create --name tpsr
conda activate tpsr
cd nesymres
pip install -e src/
pip install -r requirements.txt
pip install lightning==1.9
```




## Run
We have created `run.sh` script to execute Transformer-based Planning for Automated Equation Discovery based on the reward defined in `reward.py` with the combination of equation's fitting accuracy and complexity. To run the script for different datasets, configure the following parameters:

|   **Parameters**  |                                              **Description**                                             |       **Example Values**       |
|:-----------------:|:--------------------------------------------------------------------------------------------------------:|:------------------------------:|
| `backbone_model`    | Backbone Pre-trained Model Type (e2e/nesymres)                                | e2e      |
| `eval_in_domain`    | Evaluate backbone pre-trained model on In-Domain dataset (Yes/No)                                | True/False      |
| `eval_mcts_in_domain`    | Evaluate TPSR on In-Domain dataset (Yes/No)                                | True/False      |
| `eval_on_pmlb`        | Evaluate backbone pre-trained model on PMLB (Yes/No)                                                                     | True/False |
| `eval_mcts_on_pmlb`    | Evaluate TPSR on PMLB (Yes/No)                                  | True/False       |
| `horizon`    | Horizon of lookahead planning (maxlen of equation tokens)                                 | 200      |
| `rollout`    | Number of rollouts ($r$) in TPSR                                 | 3      |
| `num_beams`    | Beam size ($b$) in TPSR's evaluation step to simulate completed equations                                | 1      |
| `width`    |  Top-k ($k$) in TPSR's expansion step to expand tree width                                | 3     |
| `no_seq_cache`    | Use sequence caching (Yes/No)                                 | False     |
| `no_prefix_cache`    | Use top-k caching (Yes/No)                                   | False     |
| `ucb_constant`    | Exploration weight in UCB                                 | 1.0     |
| `uct_alg`    | UCT algorithm $\in$ {uct, p_uct, var_p_uct}                                 | uct     |
| `max_input_points`    | Maximum input points observed by pre-trained model ($N$)                                 | 200     |
| `max_number_bags`    | Maximum number of bags for input points ($B$)                                | 10     |
| `pmlb_data_type`    | PMLB data group $\in$ {feynman, strogatz, black-box}                                 | feynman     |
| `target_noise`    | Target noise added to y_to_fit in PMLB                                | 0.0     |
| `beam_type`    | Decoding type for pre-trained models $\in$ {search, sampling}                                | sampling     |
| `beam_size`    | Decoding size ($s$) for pre-trained models (beam size, or sampling size)                                | 10     |
| `n_trees_to_refine`    | Number of refinements in decodings $\in$ {1,..., $s$ }                                | 10     |
| `prediction_sigmas`    | Sigmas of extrapolation eval data sampling (In-domain)                                | 1,2,4,8,16     |
| `eval_input_length_modulo`    | Number of eval points (In-domain). Set to 50 yields $N_{test}=[50,100,150,200]$ per extrapolation range.                               | 50     |


## Run - PMLB Datasets (Feynman/ Strogatz/ Blackbox)
Pre-trained E2E Model (Sampling / Beam Search):
```
python run.py --eval_on_pmlb True \
                   --pmlb_data_type feynman \
                   --target_noise 0.0 \
                   --beam_type sampling \ # or search
                   --beam_size 10 \
                   --n_trees_to_refine 10 \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True
```

Transformer-based Planning with E2E Backbone:
```
python run.py --eval_mcts_on_pmlb True \
                   --pmlb_data_type feynman \
                   --target_noise 0.0 \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --max_number_bags 10 \
                   --save_results True
```
For running the code on Strogatz or Black-box datasets, simply adjust the `pmlb_data_type` parameter to either `strogatz` or `blackbox`. The commands provided above are set for the Feynman datasets. You can also modify the `target_noise` and other parameters to suit your experiments. Running each command saves the results for all datasets and metrics in a `.csv` file. 




## Run - In-Domain Datasets
**In-Domain** datasets are generated, following the validation data gneration protocol suggested in [E2E](https://arxiv.org/pdf/2204.10532.pdf). For details, refer to the `generate_datapoints` function [here](./symbolicregression/envs/generators.py). You can also modify data generation parameters [here](./symbolicregression/envs/environment.py). For example, you can adjust parameters like `prediction_sigmas` to control extrapolation. A sigma `1` aligns with the training data range, while `>1` is for extrapolation ranges.  The In-domain validation datasets are generated on-the-fly. For consistent evaluations across models, consider setting a fixed seed.


Pre-trained E2E Model (Sampling / Beam Search):
```
python run.py --eval_in_domain True \
                   --beam_type sampling \ # or search
                   --beam_size 10 \
                   --n_trees_to_refine 10 \
                   --max_input_points 200 \
                   --eval_input_length_modulo 50 \
                   --prediction_sigmas 1,2,4,8,16 \
                   --save_results True
```

Transformer-based Planning with E2E Backbone:
```
python run.py --eval_mcts_in_domain True \
                   --lam 0.1 \
                   --horizon 200 \
                   --width 3 \
                   --num_beams 1 \
                   --rollout 3 \
                   --no_seq_cache False \
                   --no_prefix_cache True \
                   --max_input_points 200 \
                   --eval_input_length_modulo 50 \
                   --prediction_sigmas 1,2,4,8,16 \
                   --save_results True \
                   --debug
```




## PMLB Inference

For simplified PMLB inference using only TPSR (no multiple method comparison), use the dedicated script:

### Single Dataset

```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/pmlb/pmlb_inference.py --dataset 1027_ESL --device cuda:0
```

### Batch Inference

无噪声 smoke，先快速验证全流程、CSV 落盘和断点续跑：

```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/pmlb/pmlb_inference.py --dataset all --device cuda:0 --dataset_limit 2 --max_rows 64 --max_input_points 32
```

默认全量无噪声批跑，结果默认写到 `experiments/pmlb/results/pmlb_batch_inference_noise_0.csv`：

```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/pmlb/pmlb_inference.py --dataset all --device cuda:0
```

带噪声示例，只给训练目标加乘性高斯噪声 `y -> y * (1 + ξ)`：

```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/pmlb/pmlb_inference.py --dataset all --device cuda:1 --noise_strength 0.1
```

按数据组筛选时可继续使用：

```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/pmlb/pmlb_inference.py --dataset all --data_type feynman --device cuda:0
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/pmlb/pmlb_inference.py --dataset all --data_type strogatz --device cuda:0
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/pmlb/pmlb_inference.py --dataset all --data_type black-box --device cuda:0
```

### Results

Results are saved to `experiments/pmlb/results/` and the default file name includes the noise strength, for example `pmlb_batch_inference_noise_0.csv` or `pmlb_batch_inference_noise_0.1.csv`.

The batch CSV columns are:
- `dataset`: Dataset name
- `status`: `success`, `failed`, or `skipped`
- `n_features`: Number of input features
- `refinement_type`: Refinement used for the final expression
- `r2`: R² score
- `rmse`: Root mean squared error
- `complexity`: Expression complexity (number of tokens)
- `seconds`: Total inference time (seconds)
- `error`: Failure or skip message
- `noise_strength`: Multiplicative Gaussian target-noise strength used for this run
- `expr`: Discovered symbolic expression

中文汇总命令：
```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_results_summary.py --input_csvs experiments/pmlb/TPSR_results/pmlb_batch_inference_noise_0.csv experiments/pmlb/TPSR_results/pmlb_batch_inference_noise_0.001.csv experiments/pmlb/TPSR_results/pmlb_batch_inference_noise_0.01.csv experiments/pmlb/TPSR_results/pmlb_batch_inference_noise_0.1.csv --output_csv experiments/pmlb/results/pmlb_results_summary.csv
```

## Synthetic Sequential-Limitation Experiment

```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/synthetic/tpsr_sequential_limitation.py --functions sin,exp,log1p --num_seeds 10 --n_points 100 --x_range -0.5 0.5 --device cuda:0

PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python experiments/synthetic/tpsr_sequential_limitation.py --functions sin --num_seeds 1 --n_points 32 --x_range -0.5 0.5 --max_input_points 32 --device cuda:0 --output_csv experiments/synthetic/results/tpsr_sequential_limitation_smoke.csv
```

## Demo
We have also included a small demo that runs TPSR with both E2E and NesymReS backbones on your dataset. You can play with it [here](./tpsr_demo.py)

### Quick Verification

**Fast test (1-2 seconds):**
```bash
# Quick test with minimal parameters
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python quick_test.py
```

**Full demo with E2E backbone (may take several minutes):**
```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python tpsr_demo.py --backbone_model e2e --no_seq_cache True --no_prefix_cache True
```

**Full demo with NeSymReS backbone:**
```bash
PYTHONPATH=nesymres/src:$PYTHONPATH .venv/bin/python tpsr_demo.py --backbone_model nesymres --no_seq_cache True --no_prefix_cache True
```

### Note on PYTHONPATH
When running with NeSymReS backbone or if you encounter `ModuleNotFoundError: No module named 'nesymres.utils'`, set the PYTHONPATH to include the nesymres source directory:
```bash
export PYTHONPATH=nesymres/src:$PYTHONPATH
```

Then use the venv python directly:
```bash
.venv/bin/python tpsr_demo.py --backbone_model e2e --no_seq_cache True --no_prefix_cache True
```


## Final Results on SRBench
Our experimental results of E2E+TPSR on SRBench datasets are provided in the [srbench_results/](./srbench_results/) directory. These results are shared to help the research community reproduce our paper's findings and serve as reference benchmarks. Each result file contains detailed performance metrics and experimental configurations used in our evaluations.



## Citation
If you find the paper or the repo helpful, please cite it with
<pre>
@article{tpsr2023,
  title={Transformer-based planning for symbolic regression},
  author={Shojaee, Parshin and Meidani, Kazem and Barati Farimani, Amir and Reddy, Chandan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={45907--45919},
  year={2023}
}
</pre>



## License 
This repository is licensed under MIT licence.



This work is built on top of other open source projects, including [End-to-End Symbolic Regression with Transformers (E2E)](https://github.com/facebookresearch/symbolicregression), [Neural Symbolic Regression that scales (NeSymReS)](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales), [Dyna Gym](https://github.com/SuReLI/dyna-gym), and [transformers](https://github.com/huggingface/transformers). We thank the original contributors of these works for open-sourcing their valuable source codes. 

## Contact Us
For any questions or issues, you are welcome to open an issue in this repo, or contact us at parshinshojaee@vt.edu, and mmeidani@andrew.cmu.edu .
