import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm, trange
import time

# ADCM
class SequenceTrainer():
    def __init__(self, 
            model, optimizer, 
            batch_size, get_batch, 
            K,
            loss_fn,
            scheduler=None, 
            eval_fns=None,
            model_path = None,
            device = 'cuda:1',
            scale = 1000,
    ):
        self.model_path = model_path
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.step_len = K
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()
        self.device = device
        self.reward_scale = scale

    def train_iteration(self, num_steps, iter_num=0,best_iter=0,best_ret=-1000,
    best_nor_ret=-100, logger=None, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        # 训练阶段
        self.model.train()
        for _ in trange(num_steps, desc="Epoch", leave=False):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        # 推理阶段
        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret, best_iter = max((best_ret,best_iter), (float(v), iter_num))
            if 'normalized_score' in k:
                if max(best_nor_ret, float(v)) != best_nor_ret:
                    model_dir = f'{self.model_path}-score-{max(best_nor_ret, float(v)):.3f}.pt'
                    torch.save(self.model.state_dict(), model_dir)
                    print(f'Best model saved to {model_dir}')  
                best_nor_ret = max(best_nor_ret, float(v))
        logs['-' * 40 + 'Best_Iteration'] = best_iter
        logs['Best_return_mean'] = best_ret
        logs['Best_normalized_score'] = best_nor_ret   

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            logger.log('=' * 80)
            logger.log(f'Iteration {iter_num}')
            for k, v in logs.items():
                logger.log(f'{k}: {v}')

        return logs,best_iter,best_ret,best_nor_ret

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(batch_size=self.batch_size, max_len=self.step_len)

        action_target = torch.clone(actions)

        action_preds, total_aux_loss = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, None,
        )
        
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target.reshape(-1, act_dim)

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )*1 + 0.05*total_aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=.5, norm_type=2)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

