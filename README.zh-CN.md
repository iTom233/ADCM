# ADCM: Advantage Decision ConvMamba

[English](README.md) | 简体中文

**Beyond Return Conditioning: Multi-Scale Sequence Modeling and Advantage-Guided Policy Routing for Offline RL** (CIKM '25)

论文链接：https://dl.acm.org/doi/abs/10.1145/3746252.3761116

详细的教程即将发布。

## 方法简介

ADCM（Advantage Decision ConvMamba）是一个基于回报条件监督学习（RCSL，Return-Conditioned Supervised Learning）的离线强化学习方法，在 Decision Transformer 范式的基础上做了两点改进：

- **多尺度序列建模（MSSM，Multi-Scale Sequence Modeling）**：将 `(RTG, state, action)` 三元组构成的嵌入序列按滑窗划分为 patch，使用 **Mamba** 建模 patch 之间的稀疏长程依赖，使用**因果卷积**建模 patch 内部稠密的马尔可夫依赖，再对两种尺度的特征做自适应融合，得到上下文感知的状态表示，从而更充分地挖掘离线数据集中多样化的行为模式。
- **优势策略引导（APG，Advantage Policy Guidance）**：在序列建模之后引入 **MoE（混合专家）** 框架。训练时，门控网络将不同状态分发给不同专家，使各专家分别拟合不同的行为分布；推理时，用预训练的 **IQL critic** 评估主动作与所有专家备选动作的价值，通过组相对优势采样选择相对更优的动作，降低对数据集中低质量行为模式的依赖。

## 模型结构

- **输入序列**：与 DT 一致，将 `(Return-to-Go, state, action)` 分别嵌入后按时间顺序拼接为长度 `3K-1` 的 token 序列（上下文长度 `K=20`，观测按数据集统计量归一化）。
- **MSSM 模块**（`src/models/seq_modeling.py`）：
  - `long_modeling`：三组错位 patch 序列分别输入三个 **Mamba2**，建模 patch 间的长程依赖；
  - `short_modeling`：RTG / state / action 三路独立的**因果一维卷积**，建模 patch 内的局部马尔可夫依赖；
  - `MSTF`：对长程与局部两路特征进行可学习加权融合。
- **MoE / APG 模块**（`src/models/moe.py`）：16 个专家网络（MLP）+ 门控网络；训练阶段按门控权重融合专家输出进行行为模仿，推理阶段额外获得全部专家的备选动作。
- **Critic**（`src/models/iql.py`）：基于 IQL（expectile 回归，τ=0.6）预训练的 TwinQ 网络，用于评估动作价值，推理阶段冻结使用。

## 源码结构

```
.
├── src/
│   ├── experiment.py                    # 训练入口：数据加载 / 序列采样 / 训练-评估主循环
│   ├── visualize_inference.py           # 加载模型权重进行推理可视化
│   ├── models/
│   │   ├── ADCM.py                      # 主模型：序列嵌入 + MSSM + MoE + 训练/推理分支
│   │   ├── seq_modeling.py              # MSSM：Mamba2 / 因果卷积 / 多尺度特征融合
│   │   ├── moe.py                       # MoE：门控网络、专家网络、负载均衡损失
│   │   └── iql.py                       # IQL 实现与 critic 预训练
│   ├── training/trainer.py              # 训练步、评估、最优模型保存
│   ├── evaluation/evaluate_episodes.py  # RTG 条件推理 rollout
│   ├── tools/                           # logger / replay buffer / 通用工具
│   └── envs/                            # 辅助环境 Reacher2d（未参与主实验）
├── data/                                # 离线数据集缓存（*.pkl）
├── critic_models/                       # 预训练 IQL critic（q_network.pt / v_network.pt）
└── save/                                # 训练日志与模型权重
```

## 环境依赖

- Python 3.8+，PyTorch（建议 CUDA），MuJoCo
- Python 依赖及本项目实测版本见 [`requirements.txt`](requirements.txt)；其中 `mujoco-py` 还需要本机安装 MuJoCo 2.0 运行时：

```bash
pip install -r requirements.txt
```

- `D4RL`（v1.1）与 `mjrl`（v1.0.0）是第三方库（未包含在本仓库中），需从官方仓库安装：

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
pip install -e D4RL
git clone https://github.com/aravindr93/mjrl.git
pip install -e mjrl
```

## 运行

实验在单张 GPU（RTX 3090）上完成。

### 1. 准备数据与 Critic

- 离线数据：`data/{env}-{dataset}-v2.pkl`，每条轨迹包含 `observations` / `actions` / `rewards` / `terminals` 等字段（仓库已附 `walker2d-medium-replay-v2.pkl` 示例）。
- Critic：默认从 `critic_models/{gym_name}/q_network.pt` 加载预训练 Q 网络；若需重新预训练，可取消 `src/experiment.py` 中 `pre_train_IQL(...)` 一行的注释（`src/models/iql.py` 中也提供了完整的 IQL 训练入口），模型将保存到 `critic_models/`。

### 2. 训练

```bash
cd src
python experiment.py --env walker2d --dataset medium-replay --lambda_param 1.0
```

主要命令行参数（详见 `src/experiment.py`）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | walker2d | 环境：hopper / halfcheetah / walker2d / maze2d |
| `--dataset` | medium-replay | 数据集：medium / medium-replay / medium-expert（maze2d 为 umaze / medium / large） |
| `--K` | 20 | 上下文长度 |
| `--batch_size` | 64 | 批大小 |
| `--embed_dim` | 128 | 嵌入维度 |
| `--lambda_param` | 1.0 | 优势采样与回报条件的权衡系数 λ |
| `--max_iters` / `--num_steps_per_iter` | 50 / 2000 | 训练迭代数与每轮训练步数 |
| `--device` | cuda:0 | 训练设备 |

训练过程中每个迭代评估一次（默认 10 个 episode），并按归一化分数保存最优模型到 `save/models/`；日志保存在 `save/{实验名}/`（`logger.log` / `progress.csv` / `variant.json`）。

### 3. 推理可视化

```bash
cd src
python visualize_inference.py --env walker2d --dataset medium-replay \
    --model_path ../save/models/walker2d-medium-replay-v2-score-84.378.pt
```

## 实验结果

- **Gym**（halfcheetah / hopper / walker2d × medium / medium-replay / medium-expert，共 9 个数据集）：平均归一化分数 **87.0**，优于 DT、CQL、IQL、CGDT、DC、DM、LSDT 等基线，在 medium / medium-replay 等次优数据集上提升尤为明显。
- **Maze2d**（umaze / medium / large）：平均归一化分数 **117.4**。

详细结果与分析请参见论文。

## 引用

```bibtex
@inproceedings{wu2025beyond,
  author    = {Wu, Kunbao and Zhu, Xinning and Qin, Yang and Wang, Tieru and Diao, Jianzhou and Hu, Zheng},
  title     = {Beyond Return Conditioning: Multi-Scale Sequence Modeling and Advantage-Guided Policy Routing for Offline RL},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year      = {2025},
  doi       = {10.1145/3746252.3761116}
}
```
