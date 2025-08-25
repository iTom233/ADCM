import gym
import numpy as np
import torch

import argparse
import pickle
import random
import d4rl
import sys
sys.path.append('../')
sys.path.append('./')
# 评估函数
from src.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
# 序列建模模型
from src.models.PMD import PatchMoEDecision
from src.training.trainer import SequenceTrainer
#行为模仿模型
from src.models.seq_models import MLPBCModel
from src.training.trainer import ActTrainer

# 加载IQL模块 
from src.models.iql import pre_train_IQL, TwinQ, ValueFunction

from tqdm import trange
import torch
from tools.logger import logger, setup_logger
import os
import pathlib

def set_seed(seed):
    """设置所有随机种子确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA is available: {cuda_available}")

# 如果CUDA可用，获取CUDA设备的数量
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_gpus}")
    # 列出所有CUDA设备及其名称
    for i in range(num_gpus):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def experiment(
        exp_prefix,
        variant,
):
    seed = variant['seed']
    set_seed(seed)
    lambda_param = variant['lambda_param']
    device = variant.get('device', 'cuda')
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        # env.seed(42)
        max_ep_len = 1000
        env_targets = [7200, 3600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000,6000]
        scale = 1000.
    elif env_name == 'walker2d':
        dversion = 2
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [10000, 5000]
        scale = 1000.
    elif env_name == 'maze2d':
        if 'open' in dataset:
            dversion = 0
        else:
            dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        if dataset == 'large':
            env_targets = [600]
            scale = 100.
        if dataset == 'medium':
            env_targets = [600]
            scale = 100.
        if dataset == 'umaze':
            env_targets = [300]
            scale = 100.
    elif env_name == 'door':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [2000, 1000, 500]
        scale = 500
    elif env_name == 'kitchen':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [300]
        scale = 100.
    elif env_name == 'pen':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 3000.
    elif env_name == 'hammer':
        dversion = 1
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 3000.
    elif env_name == 'antmaze':
        dversion = 0
        gym_name = f'{env_name}-{dataset}-v{dversion}'
        env = gym.make(gym_name)
        max_ep_len = 1000
        env_targets = [1.]
        scale = 1.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations
    if not os.path.exists(os.path.join(variant['result_path'], exp_prefix)):
        pathlib.Path(
        variant['result_path'] +
        exp_prefix).mkdir(
        parents=True,
        exist_ok=True)
    setup_logger(exp_prefix, variant=variant, log_dir=os.path.join(variant['result_path'], exp_prefix))

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'C:/Users/24737/Desktop/fsdownload/ADCM_0806/data/{env_name}-{dataset}-v{+dversion}.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0
        else:
            if 'maze2d' not in env_name and 'ant' not in env_name:
                path['rewards'][-1] -= 200
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    logger.log('=' * 50)
    logger.log(f'Starting new experiment: {env_name} {dataset}')
    logger.log(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    logger.log(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    logger.log(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    logger.log('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask, mc_rtg = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1) 
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in trange(num_eval_episodes,desc="Evaluation", leave=False):
                with torch.no_grad():
                    if model_type == 'pmd':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device= device
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            result = {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_detail': returns,
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_detail': lengths,
                f'target_{target_rew}_length_std': np.std(lengths),
            }
            if env_name == 'hopper':
                result[f'target_{target_rew}_normalized_score'] =  env.get_normalized_score(np.mean(returns))*100
            if env_name == 'walker2d':
                result[f'target_{target_rew}_normalized_score'] = ((np.mean(returns)-(1.629008)) / (4592.3-(1.629008)))*100
            if env_name == 'halfcheetah':
                result[f'target_{target_rew}_normalized_score'] = ((np.mean(returns)-(-280.178953)) / (12135.0-(-280.178953)))*100
            if env_name == 'antmaze':
                result[f'target_{target_rew}_normalized_score'] = ((np.mean(returns)-(0.0)) / (1.0-(0.0)))*100
            if env_name == 'door':
                result[f'target_{target_rew}_normalized_score'] = ((np.mean(returns)-(-56.512833)) / (2880.5693087298737-(-56.512833)))*100
            if env_name == 'pen':
                result[f'target_{target_rew}_normalized_score'] = ((np.mean(returns)-(96.262799)) / (3076.8331017826877-(96.262799)))*100
            if env_name == 'hammer':
                result[f'target_{target_rew}_normalized_score'] = ((np.mean(returns)-(-274.856578)) / (12794.134825156867-(-274.856578)))*100
            if env_name == 'kitchen':
                result[f'target_{target_rew}_normalized_score'] = ((np.mean(returns)-(0.0)) / (4.0-(0.0)))*100
            if env_name == 'maze2d':
                result[f'target_{target_rew}_normalized_score'] = env.get_normalized_score(np.mean(returns))*100
            return result
        return fn
    
    #用IQL预训练Q和V
    #从新开始预训练
    # Q_fun, V_fun = pre_train_IQL(device=device, env=gym_name, seed=123, max_timesteps=int(2.5e5), eval_freq=int(5e4))
    #可以直接加载预训练模型
    Q_fun = TwinQ(state_dim, act_dim, hidden_dim=256, n_hidden=2).to(device=device)
    V_fun = ValueFunction(state_dim, hidden_dim=256, n_hidden=2).to(device=device)
    Q_fun.load_state_dict(torch.load('../critic_models/' + gym_name + '/q_network.pt', map_location=device))
    V_fun.load_state_dict(torch.load('../critic_models/' + gym_name + '/v_network.pt', map_location=device))
    Q_fun.eval()
    V_fun.eval()


    if model_type == 'pmd':
        # 需要修改DecisionTransformer模型
        model = PatchMoEDecision(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            Q_fun=Q_fun,
            V_fun=V_fun,
            lambda_param=lambda_param,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            device=device,
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型结构:")
    print(model)

    if model_type == 'pmd':
        trainer = SequenceTrainer(
            model=model,
            # dataset=dataset_hhh,
            # dataset_nm = args.dataset_nm,
            optimizer=optimizer,
            batch_size=batch_size,
            K=K,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(torch.mean((a_hat - a)**2)),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            device = device,
            scale = scale,
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            K=K,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    #开始实验
    best_iter = -1
    best_ret = -10000
    best_nor_ret = -1000
    for iter in trange(variant['max_iters'], desc="Training"):
        outputs,best_iter,best_ret,best_nor_ret = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, best_iter=best_iter,best_ret=best_ret,
    best_nor_ret=best_nor_ret, logger=logger, print_logs=True)
    return best_nor_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='walker2d') #maze2d, kitchen,hopper, walker2d, halfcheetah,hammer,door,pen,antmaze
    parser.add_argument('--dataset', type=str, default='medium-replay')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--version', type=int, default=2)
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='pmd')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=50)    
    parser.add_argument('--num_steps_per_iter', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--result_path', type=str, default='../save/')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--lambda_param', type=float, default=1.0)
    args = parser.parse_args()
    best_nor_ret = experiment('gym-experiment', variant=vars(args))
