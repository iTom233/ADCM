# ADCM: Advantage Decision ConvMamba

English | [简体中文](README.zh-CN.md)

**Beyond Return Conditioning: Multi-Scale Sequence Modeling and Advantage-Guided Policy Routing for Offline RL** (CIKM '25)

Paper: https://dl.acm.org/doi/abs/10.1145/3746252.3761116

A detailed tutorial is coming soon.

## Method Overview

ADCM (Advantage Decision ConvMamba) is an offline reinforcement learning method based on return-conditioned supervised learning (RCSL). Built upon the Decision Transformer paradigm, it introduces two key improvements:

- **Multi-Scale Sequence Modeling (MSSM)**: The embedded sequence of `(RTG, state, action)` triplets is organized into patches via a sliding window. **Mamba** models sparse long-range dependencies across patches, while **causal convolutions** capture dense Markovian dependencies within each patch. The two scales of features are then adaptively fused into context-aware state representations, enabling more effective discovery of diverse behavioral patterns in offline datasets.
- **Advantage Policy Guidance (APG)**: A **Mixture-of-Experts (MoE)** framework is introduced after sequence modeling. During training, the gating network dispatches different states to different experts, so that each expert specializes in fitting a distinct behavior distribution. During inference, a pre-trained **IQL critic** evaluates the main action together with all alternative actions proposed by the experts, and selects the relatively advantageous action via group-relative advantage sampling, reducing the reliance on low-quality behavioral patterns in the dataset.

## Model

- **Input sequence**: Following DT, `(Return-to-Go, state, action)` are embedded separately and concatenated in temporal order into a token sequence of length `3K-1` (context length `K=20`; observations are normalized with dataset statistics).
- **MSSM** (`src/models/seq_modeling.py`):
  - `long_modeling`: three groups of shifted patch sequences are fed into three **Mamba2** blocks to capture inter-patch long-range dependencies;
  - `short_modeling`: three independent **causal 1D convolutions** over RTG / state / action channels capture intra-patch local Markovian dependencies;
  - `MSTF`: learnable weighted fusion of the long-range and local feature branches.
- **MoE / APG** (`src/models/moe.py`): 16 expert networks (MLPs) + a gating network; during training the expert outputs are combined by gating weights for behavior imitation, while during inference the actions of all experts are additionally collected as alternatives.
- **Critic** (`src/models/iql.py`): a TwinQ network pre-trained with IQL (expectile regression, τ=0.6), used to evaluate action values and kept frozen during inference.

## Code Structure

```
.
├── src/
│   ├── experiment.py                    # Training entry: data loading / sequence sampling / train-eval loop
│   ├── visualize_inference.py           # Load a trained model for inference visualization
│   ├── models/
│   │   ├── ADCM.py                      # Main model: sequence embedding + MSSM + MoE + train/inference branches
│   │   ├── seq_modeling.py              # MSSM: Mamba2 / causal convolution / multi-scale feature fusion
│   │   ├── moe.py                       # MoE: gating network, expert networks, load-balancing loss
│   │   └── iql.py                       # IQL implementation and critic pre-training
│   ├── training/trainer.py              # Training step, evaluation, best-model saving
│   ├── evaluation/evaluate_episodes.py  # RTG-conditioned inference rollout
│   ├── tools/                           # logger / replay buffer / utilities
│   └── envs/                            # Auxiliary environment Reacher2d (not used in main experiments)
├── data/                                # Cached offline datasets (*.pkl)
├── critic_models/                       # Pre-trained IQL critic (q_network.pt / v_network.pt)
└── save/                                # Training logs and model checkpoints
```

## Requirements

- Python 3.8+, PyTorch (CUDA recommended), MuJoCo
- Python dependencies with the versions tested in this project are listed in [`requirements.txt`](requirements.txt); `mujoco-py` additionally requires a local MuJoCo 2.0 installation:

```bash
pip install -r requirements.txt
```

- `D4RL` (v1.1) and `mjrl` (v1.0.0) are third-party libraries (not tracked in this repository) and need to be installed from their official repositories:

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
pip install -e D4RL
git clone https://github.com/aravindr93/mjrl.git
pip install -e mjrl
```

## Getting Started

All experiments were conducted on a single GPU (RTX 3090).

### 1. Data and Critic Preparation

- Offline data: `data/{env}-{dataset}-v2.pkl`, where each trajectory contains fields such as `observations` / `actions` / `rewards` / `terminals` (an example `walker2d-medium-replay-v2.pkl` is included).
- Critic: by default the pre-trained Q network is loaded from `critic_models/{gym_name}/q_network.pt`. To pre-train it from scratch, uncomment the `pre_train_IQL(...)` line in `src/experiment.py` (a full IQL training entry is also provided in `src/models/iql.py`); the models will be saved to `critic_models/`.

### 2. Training

```bash
cd src
python experiment.py --env walker2d --dataset medium-replay --lambda_param 1.0
```

Main command-line arguments (see `src/experiment.py` for details):

| Argument | Default | Description |
| --- | --- | --- |
| `--env` | walker2d | Environment: hopper / halfcheetah / walker2d / maze2d |
| `--dataset` | medium-replay | Dataset: medium / medium-replay / medium-expert (umaze / medium / large for maze2d) |
| `--K` | 20 | Context length |
| `--batch_size` | 64 | Batch size |
| `--embed_dim` | 128 | Embedding dimension |
| `--lambda_param` | 1.0 | Trade-off coefficient λ between advantage sampling and return conditioning |
| `--max_iters` / `--num_steps_per_iter` | 50 / 2000 | Number of iterations and training steps per iteration |
| `--device` | cuda:0 | Training device |

The model is evaluated once per iteration (10 episodes by default), and the best model is saved to `save/models/` according to the normalized score. Logs are saved in `save/{experiment_name}/` (`logger.log` / `progress.csv` / `variant.json`).

### 3. Inference Visualization

```bash
cd src
python visualize_inference.py --env walker2d --dataset medium-replay \
    --model_path ../save/models/walker2d-medium-replay-v2-score-84.378.pt
```

## Results

- **Gym** (halfcheetah / hopper / walker2d × medium / medium-replay / medium-expert, 9 datasets): average normalized score **87.0**, outperforming baselines such as DT, CQL, IQL, CGDT, DC, DM, and LSDT, with particularly strong improvements on suboptimal datasets (medium / medium-replay).
- **Maze2d** (umaze / medium / large): average normalized score **117.4**.

Please refer to the paper for detailed results and analysis.

## Citation

```bibtex
@inproceedings{wu2025beyond,
  author    = {Wu, Kunbao and Zhu, Xinning and Qin, Yang and Wang, Tieru and Diao, Jianzhou and Hu, Zheng},
  title     = {Beyond Return Conditioning: Multi-Scale Sequence Modeling and Advantage-Guided Policy Routing for Offline RL},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year      = {2025},
  doi       = {10.1145/3746252.3761116}
}
```
