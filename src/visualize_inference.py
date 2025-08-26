import gym
import numpy as np
import torch
import argparse
import pickle
import os
import sys
import time
from pathlib import Path
import d4rl
import mujoco
from PIL import Image
import io

# 添加路径
sys.path.append('./')
sys.path.append('../')

# 导入必要的模块
from models.ADCM import Advantage_Decision_ConvMamba
from src.models.iql import TwinQ
from src.evaluation.evaluate_episodes import  evaluate_episode_rtg

def set_seed(seed):
    """设置随机种子确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_env(env_name, dataset, device, lambda_param=1.0, weight_path=None):
    """加载模型和环境"""
    
    # 环境配置
    if env_name == 'hopper':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        max_ep_len = 1000
        env_targets = [7200, 3600]
        scale = 1000.
    elif env_name == 'halfcheetah':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        max_ep_len = 1000
        env_targets = [10000, 5000]
        scale = 1000.
    elif env_name == 'maze2d':
        if 'open' in dataset:
            dversion = 0
        else:
            dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        max_ep_len = 1000
        if dataset == 'large':
            env_targets = [600]
            scale = 100.
        elif dataset == 'medium':
            env_targets = [600]
            scale = 100.
        elif dataset == 'umaze':
            env_targets = [300]
            scale = 100.
    else:
        raise NotImplementedError(f"Environment {env_name} not implemented")
    
    # 创建环境（启用渲染）
    env = gym.make(gym_name)
    
    # 获取环境维度
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # 加载数据集以获取状态统计信息
    dataset_path = f'../data/{env_name}-{dataset}-v{dversion}.pkl'
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}, using default normalization")
        state_mean, state_std = 0., 1.
    else:
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        states = np.concatenate([path['observations'] for path in trajectories], axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    
    # 加载预训练的Q和V函数
    Q_fun = TwinQ(state_dim, act_dim, hidden_dim=256, n_hidden=2).to(device=device)
    
    # 创建模型
    model = Advantage_Decision_ConvMamba(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20,  # 默认K值
        Q_fun=Q_fun,
        lambda_param=lambda_param,
        max_ep_len=max_ep_len,
        hidden_size=128,  # 默认embed_dim
        n_layer=1,
        n_inner=512,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        device=device,
    )
    # 加载权重到对应设备
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    model = model.to(device=device)
    model.eval()
    
    return env, model, state_dim, act_dim, max_ep_len, scale, state_mean, state_std, env_targets

def visualize_episode(
    env, 
    model, 
    state_dim, 
    act_dim, 
    max_ep_len, 
    scale, 
    state_mean, 
    state_std, 
    device, 
    target_return=None,
    mode='normal',
):
    """可视化单个episode的推理过程"""
    
    print(f"Starting visualization with target return: {target_return}")
    
    # 设置模型为评估模式并移动到设备
    model.eval()
    model.to(device=device)
    
    # 将state_mean和state_std转换为GPU张量（与evaluate_episode_rtg对齐）
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    
    # 重置环境
    state = env.reset()
    
    # 初始化历史记录（与evaluate_episode_rtg对齐）
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    
    # ADCM模型需要target_return和timesteps（与evaluate_episode_rtg对齐）
    ep_return = target_return / scale
    target_return_tensor = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    target_reward = torch.tensor([2], device=device)
    
    episode_return = 0
    episode_length = 0
    
    # 开始推理循环
    for t in range(max_ep_len):
        # print(f"Step {t}: State shape: {states.shape}, Actions shape: {actions.shape}")
        
        # 添加padding（与evaluate_episode_rtg对齐）
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, target_reward])
        
        # 获取动作（与evaluate_episode_rtg对齐）
        with torch.no_grad():
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return_tensor.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
        
        actions[-1] = action
        action_np = action.detach().cpu().numpy()
        
        # print(f"Step {t}: Action: {action_np}")
        
        # 执行动作
        state, reward, done, info = env.step(action_np)
        
        # 更新target_reward（与evaluate_episode_rtg对齐）
        target_reward[0] = torch.max(target_reward[0], torch.tensor(reward, device=target_reward.device))
        
        # 更新历史记录
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        
        # 更新target_return和timesteps（与evaluate_episode_rtg对齐）
        if mode != 'delayed':
            pred_return = target_return_tensor[0, -1] - (reward / scale)
        else:
            pred_return = target_return_tensor[0, -1]
        
        target_return_tensor = torch.cat([target_return_tensor, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
        
        episode_return += reward
        episode_length += 1
        
        # print(f"Step {t}: Reward: {reward:.3f}, Total Return: {episode_return:.3f}")
        
        # # 在控制台显示当前状态（确保信息可见）
        # print(f"  📊 Episode Return: {episode_return:.2f} | Step: {t} | Current Reward: {reward:.2f}")

        # 渲染环境
        # env.render()
    print(f"Final episode return: {episode_return:.3f}")
    return episode_return, episode_length

def main():
    parser = argparse.ArgumentParser(description='Visualize trained model inference')
    parser.add_argument('--env', type=str, default='walker2d', 
                       help='Environment name (walker2d, hopper, halfcheetah, maze2d, antmaze)')
    parser.add_argument('--dataset', type=str, default='medium-replay', 
                       help='Dataset type (medium, medium-replay, medium-expert, expert)')
    parser.add_argument('--model_type', type=str, default='pmd', 
                       help='Model type (pmd, bc)')
    parser.add_argument('--model_path', type=str, default='../save/models/walker2d-medium-replay-v2-score-84.378.pt', 
                       help='Model path')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda, cpu)')
    parser.add_argument('--seed', type=int, default=123, 
                       help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=1, 
                       help='Number of episodes to visualize')
    parser.add_argument('--lambda_param', type=float, default=1.0, 
                       help='Lambda parameter for PMD model')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    try:
        # 加载模型和环境
        env, model, state_dim, act_dim, max_ep_len, scale, state_mean, state_std, env_targets = load_model_and_env(
            args.env, args.dataset, args.device, args.lambda_param, args.model_path
        )
        
        print(f"Successfully loaded {args.model_type} model for {args.env}-{args.dataset}")
        print(f"State dim: {state_dim}, Action dim: {act_dim}")
        print(f"Environment targets: {env_targets}")
        
        # 可视化多个episode
        for episode in range(args.num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1}/{args.num_episodes}")
            print(f"{'='*50}")
            
            # 选择目标回报
            target_return = 5000    
            
            # 可视化推理
            episode_return, episode_length = visualize_episode(
                env, model, state_dim, act_dim, max_ep_len, scale, 
                state_mean, state_std, args.device, target_return, 'normal',
            )
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Length: {episode_length}")
            print(f"  Return: {episode_return:.3f}")
            if target_return:
                print(f"  Target: {target_return}")
                print(f"  Performance: {((episode_return-(1.629008)) / (4592.3-(1.629008)))*100:.1f}%")
            
            # 等待用户输入继续下一个episode
            if episode < args.num_episodes - 1:
                input("\nPress Enter to continue to next episode...")
        
        print("\nVisualization completed!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭环境
        if 'env' in locals():
            env.close()

if __name__ == '__main__':
    main()
