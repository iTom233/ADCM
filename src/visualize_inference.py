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
from src.models.PMD import PatchMoEDecision
from src.models.seq_models import MLPBCModel
from src.models.iql import TwinQ, ValueFunction
from src.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg

def set_seed(seed):
    """设置随机种子确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_env(env_name, dataset, model_type, device, lambda_param=1.0, weight_path=None):
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
    elif env_name == 'antmaze':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        max_ep_len = 1000
        env_targets = [1.]
        scale = 1.
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
    V_fun = ValueFunction(state_dim, hidden_dim=256, n_hidden=2).to(device=device)
    
    # critic_model_path = f'../critic_models/{gym_name}'
    # if os.path.exists(f'{critic_model_path}/q_network.pt'):
    #     Q_fun.load_state_dict(torch.load(f'{critic_model_path}/q_network.pt', map_location=device))
    #     V_fun.load_state_dict(torch.load(f'{critic_model_path}/v_network.pt', map_location=device))
    #     print(f"Loaded critic models from {critic_model_path}")
    # else:
    #     print(f"Warning: Critic models not found at {critic_model_path}")
    
    # Q_fun.eval()
    # V_fun.eval()
    
    # 创建模型
    if model_type == 'pmd':
        model = PatchMoEDecision(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=20,  # 默认K值
            Q_fun=Q_fun,
            V_fun=V_fun,
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
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=20,
            hidden_size=128,
            n_layer=1,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
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
    model_type='pmd',
    mode='normal',
    save_video=False,
    video_path='inference_video.mp4',
    save_gif=True,
    gif_path='../save/git/walker2d-medium-replay-v2/inference.gif'
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
    
    if model_type == 'pmd' and target_return is not None:
        # PMD模型需要target_return和timesteps（与evaluate_episode_rtg对齐）
        ep_return = target_return / scale
        print(f'ep_return: {ep_return}')
        target_return_tensor = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        target_reward = torch.tensor([2], device=device)
    
    episode_return = 0
    episode_length = 0

    # 初始化 GIF 帧列表
    frames = []
    
    # 开始推理循环
    for t in range(max_ep_len):
        print(f"Step {t}: State shape: {states.shape}, Actions shape: {actions.shape}")
        
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
        
        print(f"Step {t}: Action: {action_np}")
        
        # 执行动作
        state, reward, done, info = env.step(action_np)
        
        # 更新target_reward（与evaluate_episode_rtg对齐）
        target_reward[0] = torch.max(target_reward[0], torch.tensor(reward, device=target_reward.device))
        
        # 更新历史记录
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        
        if model_type == 'pmd' and target_return is not None:
            # 更新target_return和timesteps（与evaluate_episode_rtg对齐）
            if mode != 'delayed':
                if env.spec.id[:7] == 'kitchen':
                    # 针对kitchen环境的特殊处理
                    pred_return = target_return_tensor[0, -1] - (reward / scale)
                else:
                    pred_return = target_return_tensor[0, -1] - (reward / scale)
            else:
                pred_return = target_return_tensor[0, -1]
            
            target_return_tensor = torch.cat([target_return_tensor, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
        
        episode_return += reward
        episode_length += 1
        
        print(f"Step {t}: Reward: {reward:.3f}, Total Return: {episode_return:.3f}")
        
        # 在控制台显示当前状态（确保信息可见）
        print(f"  📊 Episode Return: {episode_return:.2f} | Step: {t} | Current Reward: {reward:.2f}")
        
        # 渲染环境
        env.render()


        # 捕获当前帧用于 GIF
        if save_gif:
            try:
                # 获取渲染图像
                if hasattr(env, 'viewer') and env.viewer is not None:
                    # 从 mujoco_py 获取图像
                    if hasattr(env.viewer, 'read_pixels'):
                        # 获取窗口实际尺寸
                        if hasattr(env.viewer, 'window') and hasattr(env.viewer.window, 'get_size'):
                            width, height = env.viewer.window.get_size()
                        else:
                            # 使用默认尺寸
                            width, height = 1920, 1080
                        
                        result = env.viewer.read_pixels(width, height)
                        if result is not None:
                            # read_pixels 返回 (rgb_array, depth_array) 元组
                            if isinstance(result, tuple) and len(result) == 2:
                                rgb_array, _ = result  # 只取第一个元素（RGB图像）
                                # 确保像素数据是正确的格式
                                if rgb_array is not None and hasattr(rgb_array, 'shape'):
                                    # 垂直翻转图像（MuJoCo 坐标系统）
                                    rgb_array = np.flipud(rgb_array)
                                    
                                    # 转换为 PIL Image
                                    img = Image.fromarray(rgb_array)
                                    frames.append(img)
                                    print(f"  Captured frame {t} for GIF (size: {width}x{height})")
                                else:
                                    print(f"  ⚠️ Invalid RGB data format")
                            else:
                                print(f"  ⚠️ Unexpected result format from read_pixels")
                        else:
                            print(f"  ⚠️ read_pixels returned None")
            except Exception as e:
                print(f"  ❌ Failed to capture frame: {e}")

        # 在渲染窗口上显示累计奖励
        try:
            if hasattr(env, 'viewer') and env.viewer is not None:
                print(f"  🔍 Viewer found: {type(env.viewer)}")
                
                # 方法1: 尝试使用 add_overlay
                if hasattr(env.viewer, 'add_overlay'):
                    try:
                        print(f"  📝 Using add_overlay method")
                        # 清除之前的覆盖
                        env.viewer.add_overlay(
                            1,  # mjGRID_TOPLEFT = 0
                            f"Episode Return: {episode_return:.2f}",
                            "",
                        )
                        env.viewer.add_overlay(
                            1,  # 右上角
                            f"Normalized Score: {((episode_return-(1.629008)) / (4592.3-(1.629008)))*100:.1f}%",
                            "",
                        )
                        env.viewer.add_overlay(
                            1,
                            f"Step: {t}",
                            "",

                        )
                        env.viewer.add_overlay(
                            1,
                            f"Current Reward: {reward:.2f}",
                            "",
                        )
                        env.viewer.add_overlay(
                            1,
                            f"Target Return-to-go: {target_return_tensor[0, -1]*scale:.2f}",
                            "",
                        )
                        
                        # 强制刷新渲染
                        if hasattr(env.viewer, 'sync'):
                            env.viewer.sync()
                            print(f"  🔄 Synced viewer")
                            
                    except Exception as e:
                        print(f"  ❌ add_overlay failed: {e}")
                        pass
                
                # 方法2: 尝试使用 set_title 显示信息
                if hasattr(env.viewer, 'set_title'):
                    try:
                        title = f"Return: {episode_return:.2f} | Step: {t} | Reward: {reward:.2f}"
                        env.viewer.set_title(title)
                        print(f"  📋 Set viewer title: {title}")
                    except Exception as e:
                        print(f"  ❌ set_title failed: {e}")
                        pass
                        
                # 方法3: 尝试使用 window 的标题
                if hasattr(env.viewer, 'window') and hasattr(env.viewer.window, 'set_title'):
                    try:
                        title = f"Return: {episode_return:.2f} | Step: {t} | Reward: {reward:.2f}"
                        env.viewer.window.set_title(title)
                        print(f"  🪟 Set window title: {title}")
                    except Exception as e:
                        print(f"  ❌ window.set_title failed: {e}")
                        pass
                        
        except Exception as e:
            # 如果所有方法都失败，继续执行
            print(f"  ❌ All overlay methods failed: {e}")
            pass
        
        # 设置相机跟随智能体（如果环境支持）
        try:
            if hasattr(env, 'viewer') and env.viewer is not None:
                # 获取智能体当前位置
                if hasattr(env, 'get_body_com'):
                    # MuJoCo环境
                    agent_pos = env.get_body_com('torso')  # 对于Walker2d，使用torso
                    if agent_pos is not None:
                        # 设置相机位置跟随智能体
                        env.viewer.cam.lookat[:3] = agent_pos
                        env.viewer.cam.distance = 5.0  # 相机距离
                        env.viewer.cam.azimuth = 105    # 相机方位角
                        env.viewer.cam.elevation = -20  # 相机仰角
                elif hasattr(env, 'sim') and hasattr(env.sim, 'get_body_xpos'):
                    # 另一种MuJoCo环境
                    agent_pos = env.sim.get_body_xpos('torso')
                    if agent_pos is not None and hasattr(env, 'viewer') and env.viewer is not None:
                        env.viewer.cam.lookat[:3] = agent_pos
                        env.viewer.cam.distance = 3.0
                        env.viewer.cam.azimuth = 45
                        env.viewer.cam.elevation = -20
        except Exception as e:
            # 如果相机设置失败，继续执行
            pass
        
        time.sleep(0.005)  # 添加延迟以便观察
        
        if done:
            print(f"Episode finished after {episode_length} steps")
            break

    # 保存 GIF
    if save_gif and frames:
        try:
            print(f"  💾 Saving GIF with {len(frames)} frames...")
            
            # 确保目录存在
            gif_dir = os.path.dirname(gif_path)
            if gif_dir and not os.path.exists(gif_dir):
                os.makedirs(gif_dir, exist_ok=True)
                print(f"  �� Created directory: {gif_dir}")
            
            # 保存为 GIF，设置合适的帧率
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=10,  # 每帧持续时间（毫秒）
                loop=0,        # 0 表示无限循环
                optimize=True  # 优化文件大小
            )
            print(f"  ✅ GIF saved to: {gif_path}")
        except Exception as e:
            print(f"  ❌ Failed to save GIF: {e}")
            # 尝试保存到当前目录
            try:
                fallback_path = f"inference_episode_{episode_length}.gif"
                frames[0].save(
                    fallback_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=10,
                    loop=0,
                    optimize=True
                )
                print(f"  ✅ GIF saved to fallback location: {fallback_path}")
            except Exception as e2:
                print(f"  ❌ Fallback save also failed: {e2}")
    
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
    parser.add_argument('--model_path', type=str, default='../save/model_weight/walker2d-medium-replay-v2/best_model_iter_6_score_87.516.pt', 
                       help='Model path')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda, cpu)')
    parser.add_argument('--seed', type=int, default=123, 
                       help='Random seed')
    parser.add_argument('--num_episodes', type=int, default=1, 
                       help='Number of episodes to visualize')
    parser.add_argument('--lambda_param', type=float, default=1.0, 
                       help='Lambda parameter for PMD model')
    parser.add_argument('--save_video', type=bool, default=False, 
                       help='Save video of inference')
    parser.add_argument('--save_gif', type=bool, default=True, 
                       help='Save gif of inference')
    
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
            args.env, args.dataset, args.model_type, args.device, args.lambda_param, args.model_path
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
            target_return = 10000    
            
            # 可视化推理
            episode_return, episode_length = visualize_episode(
                env, model, state_dim, act_dim, max_ep_len, scale, 
                state_mean, state_std, args.device, target_return, 
                args.model_type, 'normal', args.save_video, args.save_gif
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
