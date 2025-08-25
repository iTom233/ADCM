import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
sys.path.append('C:/Users/24737/Desktop/fsdownload/ADCM_0806/src/models')
import os
from typing import List, Optional, Tuple
import copy

# 导入结构性序列建模模块
from .seq_models import PatchAttention

#导入混合专家模块
from st_moe import MoE,SparseMoEBlock

# Patch-structured Sequence Modeling and Mixture of Experts-guided Advantage Policy model
class PMD(nn.Module):
    def __init__(self, embed_dim=128, config=None, scale=False, device=None):
        super().__init__()
        self.device = device 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.pruned_heads = set()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(self.embed_dim)
        self.patch_atta = PatchAttention(embed_dim=self.embed_dim, num_heads=1, patch_size=6, device=self.device)
        # self.patch_atta_2 = PatchAttention(embed_dim=self.embed_dim, num_heads=1, patch_size=6, device=self.device)
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)
    def forward(
            self,
            hidden_states,
            output_attentions=False,
    ):
        #在这里面进行多尺度序列建模（patch内和patch间的依赖关系建模，然后SSM、卷积、attention 同时使用
        patch_atta_out = self.patch_atta(hidden_states)
        output = self.linear(patch_atta_out) 
        output = self.resid_dropout(output)
        outputs = output
        return outputs


class PatchMoEDecision(nn.Module):

    """
    This model uses PMD to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            lambda_param=1.0,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            device = 'cuda:0',
            Q_fun=None,
            V_fun=None,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.device = device
        self.lambda_param = lambda_param

        # 结构化序列建模
        self.PMD = PMD(embed_dim=hidden_size, scale=1000, device=self.device)

        # wkb 优势策略引导 
        self.moe = MoE(
            dim = self.hidden_size,
            length = 60,
            num_experts = 16,               # increase the experts (# parameters) of your model without increasing computation
            gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval = 0.2,
            capacity_factor_train = 1.2,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef = 1e-1,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef = 1e-2,      # loss weight for router z-loss
        )
        self.moe.to(device=self.device)
        # self.moe_block = SparseMoEBlock(
        #     self.moe,
        #     add_ff_before = True,
        #     add_ff_after = True
        # )
        # self.moe_block.to(device=self.device) 

        #嵌入
        # self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.rmse = nn.MSELoss()
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # 这里考虑用IQL预训练Q和上界价值函数V，帮助评价策略质量
        self.Q_fun = Q_fun
        # self.V_fun = V_fun

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        # 位置编码（考虑不用，这种位置编码会淹没原始模态的特征信息）
        # time_embeddings = self.embed_timestep(timesteps)
        # state_embeddings = state_embeddings  + time_embeddings
        # action_embeddings = action_embeddings  + time_embeddings
        # returns_embeddings = returns_embeddings + time_embeddings

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # 结构化序列建模，获取上下文感知的状态表示
        PDM_outputs = self.PMD(
            hidden_states=stacked_inputs,
        )
        # action_preds = self.predict_action(PDM_outputs[:,1::3])
        # total_aux_loss = 0

        # 混合专家引导，保证输出优势策略动作
        PDM_outputs,PDM_outputs_repeat, total_aux_loss = self.moe(PDM_outputs,rewards=rewards, hidden_states=stacked_inputs)
        if PDM_outputs.shape[0] > 1 : 
            # 训练阶段，则只进行条件行为模仿
            action_preds = self.predict_action(PDM_outputs[:,1::3])  # predict next action given state
        else :
            # 推理阶段，需要对MOE的专家行为进行评价
            action_preds = self.predict_action(PDM_outputs[:,1::3])
            action_preds_repeat = self.predict_action(PDM_outputs_repeat[:,:,1::3],)
            states_repeat = states.unsqueeze(1).repeat(1, 16, 1, 1)
            # v = self.V_fun(states[:,-1]) 
            q = self.Q_fun(states[:,-1],action_preds[:,-1])
            q_repeat = self.Q_fun(states_repeat[:,:,-1], action_preds_repeat[:,:,-1])

            # 置信度参数计算
            miu = q_repeat.mean()
            sigma = q_repeat.std()  # 修正为q_repeat的标准差（原sita计算可能存在逻辑问题）
            confidence_lower = miu - 3*sigma  # 3σ置信下限[1,4](@ref)
            confidence_upper = miu + 3*sigma  # 3σ置信上限
            # 2. V值是否在置信区间外
            # v_in_confidence = (v[0] >= confidence_lower) & (v[0] <= confidence_upper)

            if q[0] >= (miu + self.lambda_param*sigma):
                # 高价值保留
                action_preds = action_preds
            else:
                # 筛选符合置信区间的备选动作
                valid_mask = (q_repeat >= confidence_lower) & (q_repeat <= confidence_upper)
                valid_q = q_repeat.masked_fill(~valid_mask, -float('inf'))  # 非置信区间置为负无穷
                # 选择置信区间内最大价值动作
                _, Q_index = torch.max(valid_q, dim=1, keepdim=False)
                action_preds = action_preds_repeat[:, Q_index[0]] 
        return action_preds, total_aux_loss

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model（rtg指导的推理决策）
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        rewards = rewards.reshape(1, -1, 1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds,total_aux_loss = self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
