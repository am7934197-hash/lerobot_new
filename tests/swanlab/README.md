# WandB/SwanLab 双后端集成测试

本目录包含用于测试 LeRobot 中 WandB 和 SwanLab 双后端集成功能的所有测试文件和文档。

## 📁 目录结构

```
tests/swanlab/
├── __init__.py                 # 包初始化
├── README.md                   # 本文件
├── BACKEND_TEST_GUIDE.md       # 详细测试指南
├── verify_code.py              # 代码验证脚本
├── quick_test.py               # 快速功能测试
├── test_swanlab_backend.py     # 基础功能测试
├── test_mock_training.py       # 模拟训练测试
├── run_backend_tests.sh        # 测试运行脚本
└── test_configs/               # 测试配置文件
    ├── wandb_config.json
    └── swanlab_config.json
```

## 🚀 快速开始

### 1. 验证代码修改

```bash
cd /mnt/data1/jizheng/vla_project/lerobot
python tests/swanlab/verify_code.py
```

### 2. 快速测试

```bash
# 测试 WandB
python tests/swanlab/quick_test.py wandb

# 测试 SwanLab
python tests/swanlab/quick_test.py swanlab

# 测试两者
python tests/swanlab/quick_test.py both
```

### 3. 完整测试

```bash
# 运行所有测试
bash tests/swanlab/run_backend_tests.sh

# 或单独运行
python tests/swanlab/test_swanlab_backend.py --backend both
python tests/swanlab/test_mock_training.py --backend both --epochs 5
```

## 📋 测试说明

### verify_code.py - 代码验证

验证代码修改的正确性，检查：
- 文件修改完整性
- 导入语句正确性
- 后端切换逻辑
- 方法实现完整性

### quick_test.py - 快速测试

快速验证后端切换功能，测试：
- Logger 初始化
- 标量指标日志
- train/eval 分离
- 字符串指标
- 自定义步长

### test_swanlab_backend.py - 基础功能测试

全面测试后端功能，包括：
- 标量指标 (int, float)
- 字符串指标
- 自定义步长跟踪
- 多模式 (train/eval)
- 视频日志 (wandb only)
- 模型 Artifact (wandb only)
- Resume 功能

### test_mock_training.py - 模拟训练测试

模拟真实训练场景，测试：
- 多 epoch 训练循环
- 实时指标跟踪
- 训练/评估分离
- 检查点保存
- RL 风格日志

## 📖 详细文档

请查看 [`BACKEND_TEST_GUIDE.md`](./BACKEND_TEST_GUIDE.md) 获取完整的测试指南和使用说明。

## 🎯 使用示例

### 使用 WandB

```python
from lerobot.configs.default import WandBConfig

wandb_config = WandBConfig(
    enable=True,
    backend="wandb",  # 或省略，默认为 wandb
    project="my_project",
    mode="offline",
)
```

### 使用 SwanLab

```python
from lerobot.configs.default import WandBConfig

wandb_config = WandBConfig(
    enable=True,
    backend="swanlab",  # 切换到 SwanLab
    project="my_project",
    mode="offline",
)
```

## ✅ 测试状态

- ✅ 代码验证通过
- ✅ 文件结构整理完成
- ✅ 测试脚本就绪

## 📝 注意事项

1. 运行测试前请确保已安装必要的依赖：
   ```bash
   pip install wandb  # 测试 WandB
   pip install swanlab  # 测试 SwanLab
   ```

2. 测试会在 `outputs/` 或临时目录创建日志文件

3. 查看 SwanLab 日志：
   ```bash
   swanlab watch outputs/swanlog
   ```

4. 查看 WandB 日志：
   ```bash
   wandb offline
   ```

---

**位置**: `tests/swanlab/`  
**测试文件**: 5 个  
**配置文件**: 2 个  
**文档**: 2 个
