# Galbot 喷涂数据转换工具

本目录包含将 Galbot 喷涂采集的 mcap 数据转换为 LeRobot v3 格式的工具脚本。

## 环境

依赖见 [`requirements.txt`](./requirements.txt)，推荐使用 conda 环境 `lerobot` 或 `mcap_convert`：

```bash
conda activate lerobot
```

## 文件说明

### 1. `convert_galbot_dir_to_lerobot_joint_all_tasks.py`（推荐：全部任务合并为单一数据集）

与脚本 2 相同的转换逻辑，但把所有任务写入**一个**数据集：

- 所有任务共享全局连续 `episode_index` / frame index，`meta/tasks.parquet` 包含全部任务（`task_index` 按发现顺序分配且稳定）。
- 输出目录维护单个 `progress.json` 断点续传：重跑跳过已完成，自动追加新增数据和新增任务。

用法：

```bash
python convert_galbot_dir_to_lerobot_joint_all_tasks.py \
    --source-root /mnt/data2/dataset/data_jizheng/galbot_spray/spray \
    --output-dir /mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3_all \
    --job-name-yaml job_name.yaml \
    --fps 30
```

关键参数与脚本 2 相同，仅输出参数为 `--output-dir`（单数据集目录）。

### 2. `convert_galbot_dir_to_lerbot_joint.py`（每个任务一个数据集，带续传）

批量转换 `spray/` 下多个任务目录，支持**断点续传**和**新增数据自动追加**。

- 数据目录命名：`{task_num}_{recording_num}/`，如 `51398_76464/`。
- 第一个数字 `task_num` 通过 [`job_name.yaml`](./job_name.yaml) 映射为任务名。
- 相同 `task_num` 的多个文件夹合并到同一个 LeRobot 数据集目录。
- 每个任务一个独立输出目录 `{output_base}/{safe_task_name}/`。
- 每个任务目录维护 `progress.json`，记录已处理/失败的文件，重跑时跳过已完成部分，只追加新增数据。
- 转换中遇到损坏的单个 mcap 会被跳过并记入 `failed_identifiers`，不影响其他文件。

用法：

```bash
python convert_galbot_dir_to_lerbot_joint.py \
    --source-root /mnt/data2/dataset/data_jizheng/galbot_spray/spray \
    --output-base /mnt/data2/dataset/data_jizheng/galbot_spray/spray_lerobotv3 \
    --job-name-yaml job_name.yaml \
    --fps 30
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--source-root` | 源数据根目录（含 `{task_num}_{rec}/*.mcap`） |
| `--output-base` | 输出根目录，每个任务在其下建独立子目录 |
| `--job-name-yaml` | 任务编号到任务名的映射 yaml |
| `--fps` | 视频帧率，默认 30 |
| `--task-filter` | 仅处理指定任务，逗号分隔多个任务名 |
| `--max-episodes-per-task` | 每个任务最多转换的 episode 数（用于快速测试） |
| `--skip-videos` | 只写数据和元数据，跳过视频编码（调试用） |
| `--repo-id-prefix` | repo_id 前缀，默认 `local/galbot_spray` |

### 3. `convert_galbot_mcap_to_lerobot_joint.py`（单任务转换）

将**单个任务目录**下的 mcap 转换为 LeRobot v3 格式，不含任务分组和续传逻辑（是 2 的参考实现）。

用法：

```bash
python convert_galbot_mcap_to_lerobot_joint.py \
    --source-root "/path/to/task_dir" \
    --output-root /path/to/output \
    --fps 30 \
    --force
```

### 4. `convert_galbot_mcap_to_lerobot_odom.py`（含里程计的单任务转换）

在单任务转换基础上，额外从 `/odom/base_link` 提取里程计位姿，作为 `observation.odom`（7 维原始位姿 `[x, y, z, qx, qy, qz, qw]`，不做转换）写入数据集。

用法与 `convert_galbot_mcap_to_lerobot_joint.py` 相同。

### 5. `filter_dataset_arms_only.py`（数据集维度/摄像头过滤）

对已生成的数据集做降维过滤，输出新目录（同名 `_arms_only`）：

- 只保留 arm + gripper 关节，`action` / `observation.state` 从 23 维切片为 **16 维**（去掉 leg 和 head）。
- 只保留 `head_right`、`left_arm`、`right_arm` 三个摄像头，移除 `head_left`。
- `videos/` 目录用**硬链接**复制（不占额外磁盘空间）。
- 自动更新 `meta/info.json` 的 shape/names 并重算 `stats.json`（`relative_action=False`，absolute stats）。

用法：

```bash
python filter_dataset_arms_only.py
```

> 源目录、保留维度、剔除摄像头等通过脚本顶部的 `SOURCE_ROOT`、`KEEP_DIM_INDICES`、`REMOVE_CAMERAS` 等常量配置。

### 6. `fix_stats_stationary_joints.py`（修复静止关节的统计量）

修复 `stats.json` 中几乎不动的关节：其 q99-q01 极小（如 0.0001）时，分位数归一化 `2*(x-q01)/(q99-q01)-1` 会把微小噪声放大到 [-1,1] 之外，主导训练 loss。脚本将 range 小于阈值的关节的 q01/q99 放宽为 `mean ± min_range/2`（默认 `--min-range 0.1`），修改前自动备份为 `stats.json.bak`。

用法：

```bash
python fix_stats_stationary_joints.py <path/to/stats.json> [--min-range 0.1] [--dry-run]
```

### 7. `recompute_relative_stats.py`（为 Delta Action 训练重算统计量）

当 `use_relative_actions=true` 时，模型实际看到的 action 是 `action - state`（相对值），与数据集基于绝对 action 算出的 stats 不一致。此脚本将数据集硬链接复制到新目录（同名 `_relative_stats_50`），在副本上以 `relative_action=True`（`relative_exclude_joints=["gripper"]`，chunk_size=50）重算 `stats.json`，原数据集不受影响。

用法：

```bash
python recompute_relative_stats.py
```

> 重算后需将训练配置 `dataset.root` 指向输出目录。

## 数据转换说明

- 状态/动作：从 `singorix/wbcs/sensor` / `singorix/wbcs/target` 提取，按关节组拼接为 23 维向量（左右臂各 7 + 左右夹爪各 1 + 腿部 5 + 头部 2）。
- 摄像头：4 路（`head_left`、`head_right`、`left_arm`、`right_arm`），以帧数最多的相机为主时间轴，其余相机/传感器/动作对齐到主时间轴，编码为 h264 mp4。
- 输出：`data/` 存储 parquet，`videos/` 存储视频，`meta/` 存储 info/stats/episodes/tasks 等元数据。