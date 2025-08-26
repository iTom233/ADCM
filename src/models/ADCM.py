import torch
import torch.nn as nn
import sys
from .seq_modeling import PatchAttention
from .moe import MoE
sys.path.append('./')

# Multi-Scale Sequence Modeling (MSSM) and Advantage Policy guiding (APG) for offfline RL
class MSSM(nn.Module):
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
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)
    def forward(
            self,
            hidden_states,
            output_attentions=False,
    ):
        # 在这里面进行多尺度序列建模(patch内和patch间的依赖关系建模)
        patch_atta_out = self.patch_atta(hidden_states)
        output = self.linear(patch_atta_out) 
        output = self.resid_dropout(output)
        outputs = output
        return outputs

class  Advantage_Decision_ConvMamba(nn.Module):
    """
    This model uses ADCM to model (Return_1, state_1, action_1, Return_2, state_2, ...) for next advantageous action generation
    """
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            lambda_param=1.0,
            num_experts = 16,
            gating_top_n = 2,
            max_length=None,
            action_tanh=True,
            device = 'cuda:0',
            Q_fun=None,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.device = device
        self.lambda_param = lambda_param

        # 多尺度序列建模模块
        self.MSSM = MSSM(embed_dim=hidden_size, scale=1000, device=self.device)

        # MOE 
        self.moe = MoE(
            dim = self.hidden_size,
            length = max_length * 3,
            num_experts = num_experts,              
            gating_top_n = num_experts,               
            threshold_train = 0.2,          
            threshold_eval = 0.2,
            capacity_factor_train = 1.2,  
            capacity_factor_eval = 2.,      
            balance_loss_coef = 1e-1,       
            router_z_loss_coef = 1e-2,      
        )
        self.moe.to(device=self.device)

        # 嵌入
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # 预测
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.rmse = nn.MSELoss()
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # Critic
        self.Q_fun = Q_fun

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # 结构化序列建模，获取上下文感知的状态表示
        PDM_outputs = self.MSSM(
            hidden_states=stacked_inputs,
        )

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
            q = self.Q_fun(states[:,-1],action_preds[:,-1])
            q_repeat = self.Q_fun(states_repeat[:,:,-1], action_preds_repeat[:,:,-1])

            # 置信度参数计算
            miu = q_repeat.mean()
            sigma = q_repeat.std()  
            confidence_lower = miu - 3*sigma  
            confidence_upper = miu + 3*sigma  

            if q[0] >= (miu + self.lambda_param*sigma):
                # 高价值保留
                action_preds = action_preds
            else:
                # 筛选符合置信区间的备选动作
                valid_mask = (q_repeat >= confidence_lower) & (q_repeat <= confidence_upper)
                valid_q = q_repeat.masked_fill(~valid_mask, -float('inf'))  
                # 选择置信区间内最大价值动作
                _, Q_index = torch.max(valid_q, dim=1, keepdim=False)
                action_preds = action_preds_repeat[:, Q_index[0]] 
        return action_preds, total_aux_loss

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # 推理决策
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
