import argparse
import numpy as np
import torch
from tqdm import tqdm
import sys
import os
sys.path.append('../')
sys.path.append('./')
from src.experiment import experiment
import time
import json
import uuid
import datetime
# 实验配置
RESULT_PATH = '../save/'
ENV = ['halfcheetah']
DATASETS = ['medium', 'medium-replay']  # 数据集列表
VERSION = int(2)
SEEDS = [42, 123, 233, 666, 999]                        # 5个随机种子
DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def main():
    # 结果存储结构：{dataset: [result1, result2,...]}
    all_results = {ds: [] for ds in DATASETS}
    for env in ENV:
        # 遍历所有数据集
        for dataset in DATASETS:
            print(f"\n\033[1m=== ENV: {env} | DATASET: {dataset} ===\033[0m")
            # 运行5个随机种子
            dataset_results = []
            for seed in tqdm(SEEDS, desc="Running seeds", ncols=80):
                parser = argparse.ArgumentParser()
                parser.add_argument('--env', type=str, default=env) #maze2d, kitchen,hopper, walker2d, halfcheetah,hammer,door,pen,antmaze
                parser.add_argument('--dataset', type=str, default=dataset)  # medium, medium-replay, medium-expert, expert
                parser.add_argument('--version', type=int, default=VERSION)
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
                parser.add_argument('--max_iters', type=int, default=10)    
                parser.add_argument('--num_steps_per_iter', type=int, default=1000)
                parser.add_argument('--device', type=str, default=DEVICE)
                parser.add_argument('--result_path', type=str, default=RESULT_PATH+'detail/')
                parser.add_argument('--seed', type=int, default=seed)
                args = parser.parse_args()
                best_nor_ret = experiment('gym-experiment', variant=vars(args))
                print(f"\nSeed {seed} => best_nor_ret: {best_nor_ret:.2f}")
                dataset_results.append(best_nor_ret)
                print('\nsleep 5s')
                time.sleep(5)
            
            # 计算统计量
            mean = np.mean(dataset_results)
            std = np.std(dataset_results)
            # 存储结果
            all_results[dataset] = {
                'values': dataset_results,
                'mean': mean,
                'std': std
            }
            # 打印当前数据集结果
            print(f"\n\033[1mResults for {args.env}-{dataset}:\033[0m")
            print(f"Seeds: {SEEDS}")
            print(f"Scores: {[f'{x:.2f}' for x in dataset_results]}")
            print(f"Mean ± Std: {mean:.2f} ± {std:.2f}")

            # 保存
            # 当前时间戳（毫秒级）
            timestamp = int(time.time() * 1000)  # 示例: 1713691205000
            # 转换为语义化日期（精确到秒）
            date_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d_%H-%M-%S")
            score_path = f"{RESULT_PATH}score/{env}-{dataset}-{VERSION}-{date_str}.json"
            os.makedirs(os.path.dirname(score_path), exist_ok=True)
            with open(score_path, 'w') as f:
                json.dump(all_results[dataset], f, indent=4)  # indent参数美化格式

        # 打印最终汇总报告
        print("\n\033[1m" + "="*50 + "\nFINAL REPORT\n" + "="*50 + "\033[0m")
        for dataset, res in all_results.items():
            print(f"{args.env}-{dataset}:")
            print(f"  Scores: {[f'{x:.2f}' for x in res['values']]}")
            print(f"  Mean: {res['mean']:.2f} | Std: {res['std']:.2f}\n")

if __name__ == "__main__":
    main()