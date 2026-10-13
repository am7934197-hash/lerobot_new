# WandB/SwanLab 后端集成测试指南

本文档介绍如何测试 LeRobot 中 WandB 和 SwanLab 后端的集成功能。

## 📋 测试概述

我们提供了两个测试脚本来全面验证后端功能：

1. **`test_swanlab_backend.py`** - 基础功能测试
2. **`test_mock_training.py`** - 模拟真实训练场景测试

## 🎯 测试覆盖的功能

### 基础功能测试 (`test_swanlab_backend.py`)

| 测试项 | 描述 | WandB | SwanLab |
|--------|------|-------|---------|
| ✅ 标量指标日志 | int, float 类型数据 | ✓ | ✓ |
| ✅ 字符串指标日志 | 文本信息记录 | ✓ | ✓ |
| ✅ 自定义步长跟踪 | RL 风格的多步长 | ✓ | ✓ |
| ✅ 多模式日志 | train/eval 模式 | ✓ | ✓ |
| ⚠️ 视频日志 | 视频文件记录 | ✓ | 跳过 |
| ⚠️ 模型 Artifact | 模型检查点上传 | ✓ | 跳过 |
| ✅ 恢复功能 | resume 训练 | ✓ | ✓ |

### 模拟训练测试 (`test_mock_training.py`)

| 测试项 | 描述 |
|--------|------|
| 📊 多 epoch 训练 | 完整的训练循环 |
| 📈 实时指标跟踪 | loss, accuracy, lr 等 |
| 🔍 训练/评估分离 | 独立的 train/eval 日志 |
| 💾 检查点保存 | 定期保存模型 |
| 🎮 RL 风格日志 | episode/training step 双步长 |

## 🚀 快速开始

### 安装依赖

```bash
# 确保已安装 lerobot
cd /mnt/data1/jizheng/vla_project/lerobot

# 安装 wandb (可选)
pip install wandb

# 安装 swanlab (可选)
pip install swanlab
```

### 运行完整测试套件

```bash
# 方式 1: 使用测试脚本（推荐）
bash tests/swanlab/run_backend_tests.sh

# 方式 2: 手动运行各个测试
# 测试基础功能
python tests/swanlab/test_swanlab_backend.py --backend both

# 测试模拟训练
python tests/swanlab/test_mock_training.py --backend both --epochs 5

# 包含 RL 风格日志测试
python tests/swanlab/test_mock_training.py --backend both --epochs 3 --test-rl
```

### 单独测试某个后端

```bash
# 只测试 WandB
python tests/swanlab/test_swanlab_backend.py --backend wandb
python tests/swanlab/test_mock_training.py --backend wandb --epochs 5

# 只测试 SwanLab
python tests/swanlab/test_swanlab_backend.py --backend swanlab
python tests/swanlab/test_mock_training.py --backend swanlab --epochs 5
```

## 📝 测试输出说明

### WandB 后端

```
outputs/
├── wandb/              # WandB 日志目录
│   └── latest-run/
│       └── run-*.wandb
└── checkpoint_*/       # 模型检查点
    └── pretrained_model/
        └── model.safetensors
```

查看方式：
- 离线模式：`wandb offline` 查看本地日志
- 在线模式：访问 https://wandb.ai 查看云端日志

### SwanLab 后端

```
outputs/
├── swanlog/            # SwanLab 日志目录
│   └── swan-*/         # 实验目录
│       └── logs/
└── checkpoint_*/       # 模型检查点
    └── pretrained_model/
        └── model.safetensors
```

查看方式：
- 本地查看：`swanlab watch outputs/swanlog`
- 在线模式：访问 https://swanlab.cn 查看云端日志

## 🔍 测试详解

### 1. 标量指标测试

测试各种数值类型的日志记录：

```python
metrics = {
    "loss": 0.5,              # float
    "accuracy": 0.95,         # float
    "epoch": 10,              # int
    "learning_rate": 1e-4,    # float
}
logger.log_dict(metrics, step=100, mode="train")
```

### 2. 字符串指标测试

记录文本信息：

```python
metrics = {
    "best_model": "checkpoint_100",
    "status": "training",
    "device": "cuda:0",
}
logger.log_dict(metrics, step=1, mode="train")
```

### 3. 自定义步长测试

模拟 RL 训练中的多步长场景：

```python
# 记录 interaction step
logger.log_dict({
    "episode_reward": 100,
    "episode_length": 50,
}, step=None, mode="train", custom_step_key="episode")

# 记录 training step
logger.log_dict({
    "policy_loss": 0.5,
    "value_loss": 0.3,
}, step=None, mode="train", custom_step_key="training_step")
```

### 4. 训练/评估分离测试

```python
# 训练指标
logger.log_dict({"loss": 0.5}, step=100, mode="train")

# 评估指标
logger.log_dict({"eval_loss": 0.4}, step=100, mode="eval")
```

### 5. 视频日志测试（仅 WandB）

```python
logger.log_video("path/to/video.mp4", step=100, mode="train")
```

SwanLab 会优雅地跳过并记录警告日志。

### 6. 模型 Artifact 测试（仅 WandB）

```python
logger.log_policy(checkpoint_dir)
```

SwanLab 会优雅地跳过并记录信息日志。

### 7. Resume 功能测试

```python
# 第一次运行
config.wandb.run_id = None
logger1 = WandBLogger(config)
run_id = config.wandb.run_id

# 恢复运行
config.wandb.run_id = run_id
config.resume = True
logger2 = WandBLogger(config)
```

## ⚙️ 配置选项

### WandBConfig 参数说明

```python
@dataclass
class WandBConfig:
    enable: bool = False              # 是否启用日志
    backend: str = "wandb"            # 后端选择：'wandb' 或 'swanlab'
    disable_artifact: bool = False    # 禁用 artifact 上传
    project: str = "lerobot"          # 项目名称
    entity: str | None = None         # 用户/组织名
    notes: str | None = None          # 项目描述
    run_id: str | None = None         # 运行 ID（用于 resume）
    mode: str | None = None           # 模式：'online', 'offline', 'disabled'
    add_tags: bool = True             # 是否添加标签
```

### 使用示例

```python
# 使用 WandB
wandb_config = WandBConfig(
    enable=True,
    backend="wandb",
    project="my_project",
    mode="online",
)

# 使用 SwanLab
wandb_config = WandBConfig(
    enable=True,
    backend="swanlab",
    project="my_project",
    mode="offline",
)
```

## 🐛 常见问题

### Q1: 提示找不到 swanlab/wandb 模块
**A:** 确保已安装相应的包：
```bash
pip install wandb
pip install swanlab
```

### Q2: SwanLab 不支持视频/artifact 会报错吗？
**A:** 不会。这些功能会被优雅地跳过，并记录相应的日志信息。

### Q3: 如何查看离线模式的日志？
**A:** 
- WandB: `wandb offline` 或访问 `wandb/` 目录
- SwanLab: `swanlab watch outputs/swanlog`

### Q4: Resume 功能不工作怎么办？
**A:** 确保：
1. 使用相同的 `output_dir`
2. 设置 `resume=True`
3. 传递正确的 `run_id`

## 📊 性能对比

| 特性 | WandB | SwanLab |
|------|-------|---------|
| 初始化速度 | 快 | 快 |
| 日志速度 | ~1ms/次 | ~1ms/次 |
| 内存占用 | 低 | 低 |
| 本地存储 | wandb/ (二进制) | swanlog/ (文本) |
| 云端同步 | 支持 | 支持 |

## 🎓 最佳实践

1. **开发阶段**: 使用 `mode="offline"` 避免上传
2. **实验阶段**: 使用 `mode="online"` 实时跟踪
3. **生产训练**: 启用 artifact 保存最佳模型
4. **RL 训练**: 使用自定义 step key 跟踪不同步长
5. **多实验对比**: 使用不同的 `project` 或 `tags` 区分

## 📚 相关文档

- [WandB 官方文档](https://docs.wandb.ai/)
- [SwanLab 官方文档](https://docs.swanlab.cn/)
- [LeRobot 训练指南](./docs/training.md)

## 🤝 贡献

如发现任何问题或有改进建议，欢迎提交 Issue 或 Pull Request！
