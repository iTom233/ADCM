import gym
import numpy as np
import torch
import sys
import os
import d4rl

# 添加路径
sys.path.append('./')
sys.path.append('../')

def quick_test():
    """快速测试环境和模型加载"""
    
    print("=== Quick Test for Model Visualization ===")
    
    # 测试环境创建
    try:
        print("Testing environment creation...")
        env = gym.make('walker2d-medium-replay-v2')
        print(f"✓ Environment created successfully: {env.spec.id}")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # 测试环境重置
        state = env.reset()
        print(f"✓ Environment reset successful, initial state shape: {state.shape}")
        
        # 测试随机动作
        action = env.action_space.sample()
        print(f"✓ Random action generated: {action.shape}")
        
        # 测试环境步进
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Next state shape: {next_state.shape}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
        # 测试渲染
        print("Testing rendering...")
        env.render()
        print("✓ Rendering successful - you should see a window")
        
        # 等待几秒观察
        import time
        print("Waiting 3 seconds to observe rendering...")
        time.sleep(3)
        
        # 关闭环境
        env.close()
        print("✓ Environment closed successfully")
        
    except Exception as e:
        print(f"✗ Error during environment test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试模型导入
    try:
        print("\nTesting model imports...")
        from src.models.PMD import PatchMoEDecision
        from src.models.iql import TwinQ, ValueFunction
        print("✓ Model imports successful")
        
        # 测试模型创建
        print("Testing model creation...")
        state_dim = 17  # Walker2d state dimension
        act_dim = 6     # Walker2d action dimension
        
        Q_fun = TwinQ(state_dim, act_dim, hidden_dim=256, n_hidden=2)
        V_fun = ValueFunction(state_dim, hidden_dim=256, n_hidden=2)
        print("✓ Q and V functions created successfully")
        
        # 检查 CUDA 是否可用
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"✓ CUDA available, using {device}")
        else:
            device = torch.device('cpu')
            print("⚠ CUDA not available, using CPU")
        
        model = PatchMoEDecision(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=20,
            Q_fun=Q_fun,
            V_fun=V_fun,
            lambda_param=1.0,
            max_ep_len=1000,
            hidden_size=128,
            n_layer=1,
            n_inner=512,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            device=device,  # 使用检测到的设备
        )
        print("✓ PMD model created successfully")
        
        # 测试权重加载
        print("\nTesting weight loading...")
        try:
            # 查找保存的权重文件
            weight_dir = "C:/Users/24737/Desktop/fsdownload/ADCM_0806/save/model_weight/walker2d-medium-replay-v2"
            if os.path.exists(weight_dir):
                weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
                if weight_files:
                    # 选择最新的权重文件
                    latest_weight = sorted(weight_files)[-1]
                    weight_path = os.path.join(weight_dir, latest_weight)
                    
                    print(f"Found weight file: {latest_weight}")
                    
                    # 加载权重到对应设备
                    model.load_state_dict(torch.load(weight_path, map_location=device))
                    print(f"✓ Weights loaded successfully from {weight_path}")
                    
                    # 打印模型参数信息
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"模型总参数数量: {total_params:,}")
                    print(f"可训练参数数量: {trainable_params:,}")
                    print(f"模型结构:")
                    print(model)

                    
                    # 设置为评估模式
                    model.eval()
                    print("✓ Model set to evaluation mode")
                else:
                    print("⚠ No weight files found in src directory")
            else:
                print("⚠ Directory not found: C:/Users/24737/Desktop/fsdownload/ADCM_0806/src")
                
        except Exception as e:
            print(f"⚠ Weight loading failed: {e}")
            print("This is okay for initial testing")
        
    except Exception as e:
        print(f"✗ Error during model test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    quick_test()
