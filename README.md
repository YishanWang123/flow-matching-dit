# Flow Matching + DiT (Diffusion Transformer)

训练 Flow Matching + DiT 模型在 ImageNet 子集上进行图像生成。

## 项目信息

本项目主要由 **Claude Code** + **GLM-4.7** 构建。

## 功能特性

- **Flow Matching**: 使用 Rectified Flow 进行图像生成
- **DiT 架构**: 基于 Transformer 的扩散模型
- **WandB 监控**: 实时跟踪 train_loss、val_loss 和评估指标
- **评估间隔**: 训练期间可配置的评估
- **ImageNet 子集**: 更小的数据集用于更快的训练/测试
- **CIFAR-10 支持**: 真实小数据集用于快速验证

## 项目结构

```
flow-matching-dit/
├── config.py          # 训练配置（eval_interval、wandb 设置等）
├── model.py           # DiT 模型实现
├── flow_matching.py   # Flow Matching 损失和采样
├── dataset.py         # ImageNet 子集数据加载器
├── train.py           # 主训练脚本，带 wandb 日志记录
└── utils.py           # 工具函数
```

## 安装

```bash
pip install -r requirements.txt
```

## 训练

### 使用 CIFAR-10 数据集（推荐用于快速验证）

```bash
python train.py --use_cifar10 --wandb_project your_project_name
```

### 使用 ImageNet 数据集

```bash
python train.py --wandb_project your_project_name
```

### 使用合成数据（测试用）

```bash
python train.py --use_synthetic --wandb_project your_project_name
```

### 恢复训练

```bash
python train.py --use_cifar10 --resume checkpoints/checkpoint_epoch_50.pt
```

## 训练配置

主要配置项（在 `config.py` 中）：

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `model_dim` | 256 | DiT 隐藏层维度 |
| `num_heads` | 8 | 注意力头数 |
| `num_layers` | 6 | Transformer 层数 |
| `batch_size` | 32 | 批次大小 |
| `learning_rate` | 0.0001 | 学习率 |
| `num_epochs` | 100 | 训练轮数 |
| `eval_interval` | 1000 | 评估间隔（步数） |
| `save_interval` | 5000 | 保存间隔（步数） |
| `log_interval` | 100 | 日志间隔（步数） |
| `image_size` | 32 | 图像尺寸 |
| `num_classes` | 1000 / 10 | 类别数（ImageNet/CIFAR-10） |

## WandB 监控

训练过程会自动记录以下指标到 WandB：

- `train_loss` - 训练损失
- `train_loss_avg` - 训练平均损失
- `val_loss` - 验证损失
- `learning_rate` - 学习率
- `samples` - 生成的样本图片

## 训练示例

使用 CIFAR-10 数据集训练 100 个 epoch 的结果：

| Epoch | 平均 Loss |
|-------|-----------|
| 1     | ~1.15     |
| 20    | 1.06      |
| 40    | 0.78      |
| 60    | 0.52      |
| 80    | 0.35      |
| 100   | 0.24      |

Loss 从 1.15 降至 0.24，模型成功收敛。

## 检查点

训练过程中会自动保存检查点到 `./checkpoints/` 目录：

```
checkpoints/
├── checkpoint_epoch_0.pt
├── checkpoint_epoch_10.pt
...
├── checkpoint_epoch_99.pt
├── final_model.pt
└── samples/
    ├── samples_step_1000.png
    ├── samples_step_2000.png
    ...
    └── samples_step_31199.png
```

## 模型架构

### DiT (Diffusion Transformer)

- **Patch Embedding**: 将图像分割成 patches 并嵌入
- **Position Embeddings**: 正弦位置编码
- **Time Embeddings**: 时间步条件
- **Class Embeddings**: 类别标签条件（可选）
- **Transformer Blocks**: 多层自注意力 + 前馈网络
- **输出层**: 预测速度场

### Flow Matching

- **Rectified Flow**: 从噪声到数据的线性轨迹
- **ODE 求解器**: Euler 和改进的 Euler 方法
- **采样步数**: 可配置的采样步数

## 依赖项

- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `wandb>=0.15.0`
- `numpy>=1.24.0`
- `tqdm>=4.65.0`
- `pillow>=9.5.0`
- `pyyaml>=6.0`

## 许可证

MIT License
