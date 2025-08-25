# 强化学习模型可视化推理指南

本目录包含了用于可视化训练好的强化学习模型推理过程的脚本。

## 文件说明

### 1. `quick_test.py` - 快速测试脚本
用于快速验证环境和模型是否能正常加载和运行。

**使用方法：**
```bash
cd src
python quick_test.py
```

**功能：**
- 测试gym环境创建和渲染
- 测试模型导入和创建
- 测试预训练权重加载
- 验证基本功能是否正常

### 2. `visualize_inference.py` - 完整可视化脚本
用于加载训练好的模型并在gym环境中进行实时推理可视化。

**使用方法：**
```bash
cd src

# 基本用法（使用默认参数）
python visualize_inference.py

# 指定环境和数据集
python visualize_inference.py --env walker2d --dataset medium

# 指定模型类型
python visualize_inference.py --model_type pmd

# 指定设备
python visualize_inference.py --device cpu

# 指定episode数量
python visualize_inference.py --num_episodes 5

# 查看所有可用参数
python visualize_inference.py --help
```

## 支持的环境

| 环境名称 | 支持的数据集 | 版本 |
|---------|-------------|------|
| `walker2d` | medium, medium-replay, medium-expert, expert | v2 |
| `hopper` | medium, medium-replay, medium-expert, expert | v2 |
| `halfcheetah` | medium, medium-replay, medium-expert, expert | v2 |
| `maze2d` | large, medium, umaze | v0/v1 |
| `antmaze` | large-diverse, medium-diverse, umaze-diverse, umaze | v0 |

## 支持的模型类型

1. **PMD (PatchMoEDecision)** - 默认模型
   - 使用混合专家和结构化序列建模
   - 需要预训练的Q和V函数

2. **BC (Behavior Cloning)** - 行为克隆模型
   - 简单的MLP模型
   - 不需要预训练的critic函数

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--env` | walker2d | 环境名称 |
| `--dataset` | medium | 数据集类型 |
| `--model_type` | pmd | 模型类型 (pmd/bc) |
| `--device` | cuda | 计算设备 (cuda/cpu) |
| `--seed` | 42 | 随机种子 |
| `--num_episodes` | 3 | 可视化的episode数量 |
| `--lambda_param` | 1.0 | PMD模型的lambda参数 |

## 使用步骤

### 步骤1：快速测试
首先运行快速测试脚本，确保环境正常：
```bash
python quick_test.py
```

### 步骤2：可视化推理
如果测试通过，运行完整可视化脚本：
```bash
python visualize_inference.py --env walker2d --dataset medium
```

### 步骤3：观察结果
- 脚本会打开一个渲染窗口显示环境
- 模型会实时进行推理并执行动作
- 控制台会显示每个步骤的详细信息
- 每个episode结束后按Enter继续下一个

## 故障排除

### 常见问题1：CUDA不可用
```bash
# 自动切换到CPU
python visualize_inference.py --device cpu
```

### 常见问题2：模型权重未找到
确保预训练权重文件存在于正确路径：
```
critic_models/
├── walker2d-medium-v2/
│   ├── q_network.pt
│   └── v_network.pt
├── hopper-medium-v2/
│   ├── q_network.pt
│   └── v_network.pt
└── ...
```

### 常见问题3：环境渲染问题
- 确保安装了正确的gym版本
- 对于MuJoCo环境，确保安装了mujoco-py
- 某些环境可能需要额外的依赖包

### 常见问题4：导入错误
检查Python路径和依赖：
```bash
# 安装依赖
pip install torch gym numpy

# 检查路径
python -c "import sys; print(sys.path)"
```

## 性能优化建议

1. **使用GPU加速**：如果有CUDA，使用`--device cuda`
2. **调整渲染速度**：在脚本中修改`time.sleep()`的值
3. **批量推理**：可以修改脚本支持批量episode推理

## 扩展功能

### 训练过程中的自动权重保存
在训练过程中，当检测到更好的归一化分数时，系统会自动保存最佳模型权重：

```python
# 在训练器的train_iteration方法中
if 'normalized_score' in k:
    best_nor_ret = max(best_nor_ret, float(v))
    if float(v) > best_nor_ret:
        # 自动保存最佳模型权重
        save_best_model_weights(model, iter_num, float(v), save_dir)
```

**自动保存的特点：**
- 实时监控：每次评估后自动检查是否达到新的最佳分数
- 智能命名：文件名包含迭代次数、分数和时间戳
- 配置记录：同时保存模型架构和参数统计信息
- 异常处理：保存失败时不会中断训练过程

### 保存模型权重
脚本支持保存训练好的模型权重：
```bash
# 保存模型权重
python visualize_inference.py --save_weights

# 指定权重保存路径
python visualize_inference.py --save_weights --weights_path ./my_models

# 同时保存视频和权重
python visualize_inference.py --save_video --save_weights
```

**保存的内容包括：**
- 主模型权重文件 (`.pt`格式)
- Q和V函数权重（如果使用PMD模型）
- 模型配置文件 (`.json`格式，包含架构信息和参数统计)
- 时间戳标识，便于版本管理

### 保存视频
脚本支持保存推理过程的视频（需要额外配置）：
```bash
python visualize_inference.py --save_video
```

### 自定义环境
可以修改脚本支持其他gym环境，只需添加相应的环境配置。

### 多模型比较
可以修改脚本同时加载多个模型进行比较推理。

## 注意事项

1. **内存使用**：大型模型可能需要较多GPU内存
2. **渲染性能**：实时渲染可能影响推理速度
3. **随机性**：设置随机种子确保结果可复现
4. **环境关闭**：脚本会自动关闭环境，但建议手动检查

## 联系支持

如果遇到问题，请检查：
1. 依赖包版本是否正确
2. 文件路径是否正确
3. 模型权重是否存在
4. 环境配置是否正确
