# PI0.5 训练指南

## 快速开始

### 基本命令

```bash
# 使用 GPU 4,5 训练
CUDA_VISIBLE_DEVICES="4,5" HF_ENDPOINT="https://hf.mirror.com" && python tests/swanlab/train_pi05_with_swanlab.py \
    --api-key lxinhZyaQbBOWbcnFbfdX \
    --project lerobot-pi05-test \
    --epochs 5 \
    --batch-size 32 \
    --max-samples 1000
```

**注意**: `HF_ENDPOINT` 镜像已在脚本内部设置，无需在命令行中指定。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-key` | None | SwanLab API key |
| `--project` | lerobot-pi05-test | SwanLab 项目名称 |
| `--entity` | None | SwanLab 工作空间名称 |
| `--epochs` | 10 | 训练轮数 |
| `--batch-size` | 8 | 批次大小 (PI0.5 需要更多显存，建议 8-16) |
| `--dataset` | lerobot/svla_so101_pickplace | 数据集 ID |
| `--device` | cuda | 设备 (cuda/cpu) |
| `--output-dir` | outputs/pi05_swanlab | 输出目录 |
| `--max-samples` | None | 最大训练样本数（用于快速测试） |

## 训练配置示例

### 1. 快速测试（1000 样本，5 个 epoch）

```bash
CUDA_VISIBLE_DEVICES="4,5" python tests/swanlab/train_pi05_with_swanlab.py \
    --api-key lxinhZyaQbBOWbcnFbfdX \
    --project lerobot-pi05-test \
    --epochs 5 \
    --batch-size 16 \
    --max-samples 1000
```

### 2. 完整训练（使用全部数据）

```bash
CUDA_VISIBLE_DEVICES="4,5" python tests/swanlab/train_pi05_with_swanlab.py \
    --api-key lxinhZyaQbBOWbcnFbfdX \
    --project lerobot-pi05-full \
    --epochs 50 \
    --batch-size 16 \
    --dataset lerobot/svla_so101_pickplace
```

### 3. 使用更小的 batch size（节省显存）

```bash
CUDA_VISIBLE_DEVICES="4" python tests/swanlab/train_pi05_with_swanlab.py \
    --api-key lxinhZyaQbBOWbcnFbfdX \
    --project lerobot-pi05-test \
    --epochs 20 \
    --batch-size 8
```

## 查看训练日志

### 本地查看
```bash
swanlab watch outputs/pi05_swanlab/swanlog
```

### 云端查看
访问：https://swanlab.cn/@jizheng/lerobot-pi05-test

## 模型保存

训练完成后，模型会保存在以下目录：
- `outputs/pi05_swanlab/best_model/` - 最佳验证模型
- `outputs/pi05_swanlab/checkpoint_epoch_X/` - 每 2 个 epoch 的检查点

## 常见问题

### 1. 网络连接问题
脚本已自动设置 `HF_ENDPOINT` 镜像，如果仍有问题，检查：
```bash
export HF_ENDPOINT="https://hf.mirror.com"
```

### 2. 显存不足
- 减小 `--batch-size`（建议 8 或更小）
- 使用 `--max-samples` 限制数据集大小
- 减少 `--epochs`

### 3. CUDA 不可用
脚本会自动回退到 CPU 模式，但训练速度会慢很多。
