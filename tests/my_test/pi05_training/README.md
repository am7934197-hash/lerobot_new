# PI05 单臂喷涂训练

基于 LeRobot PI05（Pi0.5）的单臂喷涂任务训练，包含数据集处理、训练配置、Checkpoint 转换等完整流程。

> 数据路径：`/mnt/data2/dataset/data_jizheng/galbot_spray/spray_0708/`

---

## 目录结构

```
pi05_training/
├── README.md                              # 本文档
├── .gitignore                             # 忽略 outputs/
├── accelerate.yaml                        # accelerate 分布式训练配置
│
├── 训练脚本
│   ├── lerobot_pi05_train.sh              # demo 级单卡训练（不推荐，已废弃）
│   ├── accelerate_pi05_train_config.sh    # [主] 绝对动作 4cam 训练（通过 accelerate.yaml 读配置）
│   ├── pi05_train_accelerate.sh           # [主] 绝对动作 4cam 训练（CLI 传参）
│   ├── pi05_train_accelerate_deltaaction.sh  # [主] Delta Action 4cam 训练
│   ├── pi05_train_accelerate_3cam.sh         # 3 摄像头绝对动作训练
│   └── pi05_train_accelerate_3cam_delta.sh   # 3 摄像头 Delta Action 训练
│
├── 训练配置文件
│   ├── pi05_test_swanlab_config.yaml          # [主] 绝对动作 4cam 完整配置
│   ├── pi05_test_swanlab_config_delta.yaml    # [主] Delta Action 4cam 配置
│   ├── pi05_3cam_config.yaml                  # 绝对动作 3cam 配置
│   └── pi05_3cam_delta_config.yaml            # Delta Action 3cam 配置
│
├── 数据处理脚本
│   ├── recompute_relative_stats.py            # 为 Delta Action 重算数据集统计量
│   ├── filter_dataset_arms_only.py             # 过滤数据集：仅保留 arm+gripper 维度
│   └── fix_stats_stationary_joints.py          # 拓宽静止关节的 q01/q99 范围
│
├── Checkpoint 工具
│   └── convert_to_bf16.py                      # float32 → bfloat16 模型转换
│
└── outputs/                                    # 训练输出（gitignore）
```

---

## 数据关节布局（23 维）

| 索引 | 关节名 | 说明 |
|------|--------|:--------------:|------|
| 0-6  | right_arm_joint1~7| 实际控制的右臂 7 关节 |
| 7    | right_arm_gripper | 右臂夹爪（0~100） |
| 8-14 | left_arm_joint1~7 | 左臂，数采时未动 |
| 15   | left_arm_gripper | 左臂夹爪（0~100） |
| 16-20| leg_joint1~5 | 腿部关节，仅轻微抖动 |
| 21-22| head_joint1~2 | 头部关节 |

---

## 数据处理流程

### 数据集之间存在派生关系

```
源数据 (50003_71684_lerobotv3)
  ├── [recompute_relative_stats.py] → _relative_stats_50   (Delta Action 统计量)
  ├── [filter_dataset_arms_only.py]  → _arms_only          (仅 arm+gripper 16 维)
  └── [fix_stats_stationary_joints.py] → 原地修改 stats.json（拓宽静止关节范围）
```

### 1. `recompute_relative_stats.py` — 重算 Delta Action 统计量

**用途**：当使用 `use_relative_actions=true` 训练时，需先用 action - state（相对值）重算数据集的 q01/q99/mean/std，使归一化与模型实际看到的分布一致。

**输出**：`50003_71684_lerobotv3_relative_stats_50`（硬链接复制，不占额外磁盘空间）。

```bash
uv run python tests/my_test/pi05_training/recompute_relative_stats.py
```

### 2. `filter_dataset_arms_only.py` — 过滤仅保留臂部关节

**用途**：将 action/state 从 23 维裁剪为 16 维（仅 right_arm + left_arm + grippers），同时移除 `head_left` 摄像头。

**输出**：`50003_71684_lerobotv3_arms_only`。

```bash
uv run python tests/my_test/pi05_training/filter_dataset_arms_only.py
```

> 注意：此脚本会创建新的 parquet 文件（非硬链接），需额外磁盘空间。

### 3. `fix_stats_stationary_joints.py` — 拓宽静止关节归一化范围

**问题**：单臂喷涂中 legs/left_arm/head 等 15 个关节几乎不运动，q99-q01 极小（~0.0001）。QUANTILES 归一化公式 `2*(x-q01)/(q99-q01)-1` 会将微小噪声（state→action 复制误差）放大到 [-1,1] 范围之外，导致这些维度的 loss 主导总 loss。

**解决**：将静止关节的 q01/q99 拓宽到 `mean ± 0.05`（range=0.1），噪声不再被放大。详见前文诊断分析。

```bash
# 预览改动
python tests/my_test/pi05_training/fix_stats_stationary_joints.py --dry-run

# 修改绝对动作数据集
python tests/my_test/pi05_training/fix_stats_stationary_joints.py \
  /mnt/data2/dataset/data_jizheng/galbot_spray/spray_0708/50003_71684_lerobotv3/meta/stats.json

# 修改 relative_stats 数据集
python tests/my_test/pi05_training/fix_stats_stationary_joints.py \
  /mnt/data2/dataset/data_jizheng/galbot_spray/spray_0708/50003_71684_lerobotv3_relative_stats_50/meta/stats.json

# 可调参数
#   --min-range 0.2    最小 range（默认 0.1）
#   --stationary-indices 8 9 ... 22   指定静止维度
```

---

## 训练配置说明

### 配置变体矩阵

| 配置文件 | 摄像头 | 动作模式 | 数据集 | 使用脚本 |
|----------|:------:|:--------:|--------|----------|
| `pi05_test_swanlab_config.yaml` | 4 (head_left/right, left_arm, right_arm) | 绝对 | `lerobotv3` | `pi05_train_accelerate.sh` |
| `pi05_test_swanlab_config_delta.yaml` | 4 | Delta | `_relative_stats_50` | `pi05_train_accelerate_deltaaction.sh` |
| `pi05_3cam_config.yaml` | 3 (无 head_left) | 绝对 | `lerobotv3` | `pi05_train_accelerate_3cam.sh` |
| `pi05_3cam_delta_config.yaml` | 3 | Delta | `_relative_stats_50` | `pi05_train_accelerate_3cam_delta.sh` |

### 关键配置参数

| 参数 | 值 | 说明 |
|------|:--:|------|
| `policy.type` | `pi05` | 策略类型 |
| `paligemma_variant` | `gemma_2b` | VLM 骨干 |
| `action_expert_variant` | `gemma_300m` | 动作专家 |
| `dtype` | `bfloat16` | 训练精度 |
| `chunk_size` | `50` | 预测动作块大小 |
| `n_action_steps` | `50` | 执行步数 |
| `normalization_mapping.ACTION` | `QUANTILES` | 动作归一化方式 |
| `optimizer_lr` | `2.5e-5` | 峰值学习率 |
| `gradient_checkpointing` | `true` | 省显存 |
| `batch_size` | `16` (4cam) / `8` (3cam) | 每卡 batch |
| `steps` | `10000` | 总训练步数 |

### 有效 Batch Size

```
有效 batch = YAML batch_size × num_processes × gradient_accumulation_steps
```

- 4cam 脚本：`16 × 2 × 2 = 64`
- 3cam 脚本：`8 × 2 × 4 = 64`

---

## 训练启动

### 环境准备

```bash
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 启动命令

```bash
# 绝对动作训练（4 摄像头）
bash pi05_train_accelerate.sh

# Delta Action 训练（4 摄像头，推荐）
bash pi05_train_accelerate_deltaaction.sh

# 3 摄像头 + Delta Action
bash pi05_train_accelerate_3cam_delta.sh

# 覆盖参数
NUM_PROCESSES=4 bash pi05_train_accelerate.sh          # 4 卡
MIXED_PRECISION=fp16 bash pi05_train_accelerate.sh     # fp16
```

### 监控

训练日志通过 SwanLab 记录，项目名 `lerobot-pi05-painting`。

---

## Checkpoint 转换

### `convert_to_bf16.py` — float32 → bfloat16

将训练好的 checkpoint 从 float32 转换为 bfloat16，体积减半。

```bash
uv run python tests/my_test/pi05_training/convert_to_bf16.py

# 自定义路径
python convert_to_bf16.py --src <src_dir> --dst <dst_dir>
```

---

## accelerate.yaml

分布式训练框架配置（DDP），关键设置：

| 字段 | 值 | 说明 |
|------|:--:|------|
| `distributed_type` | `MULTI_GPU` | DDP 多卡 |
| `mixed_precision` | `bf16` | bfloat16 混合精度 |
| `num_processes` | `1` | 默认值，脚本通过 CLI 覆盖为 2 |

脚本通过 `--config_file accelerate.yaml` 读取此配置。部分脚本也用 CLI 直接传参（`--num_processes --multi_gpu`）覆盖。

---

## 常见问题

**Q: 为什么 loss 从 0.001（arm）变成 0.1（全身）？**  
A: 静止关节的 q99-q01 范围极小，QUANTILES 归一化放大了 state→action 复制时的微小噪声。处理方法：运行 `fix_stats_stationary_joints.py` 拓宽静止关节的 q01/q99 范围。

**Q: 为什么要用 Delta Action？**  
A: Delta Action（相对动作）预测的是「当前状态到目标动作的偏移量」，比直接预测绝对位置更容易学习，收敛更快。

**Q: 3 摄像头 vs 4 摄像头的区别？**  
A: 原始数据有 4 个摄像头（head_left, head_right, left_arm, right_arm）。`head_left` 和 `head_right` 是头部双目，实际训练中通过 `input_features` 显式指定可排除 `head_left`。
