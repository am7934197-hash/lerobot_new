# LeRobot（Galbot PI05 喷涂版）

基于 [HuggingFace LeRobot](https://github.com/huggingface/lerobot) 的定制版本，用于 **Galbot G1 机器人 PI05（Pi0.5）喷涂任务**：数据转换 → 模型训练 → gRPC 部署。

> 当前分支 `swanlab`，基于官方 `main` 分支（VLA-JEPA）定制。

## 相对官方 main 的主要修改

| 修改点 | 位置 | 说明 |
|--------|------|------|
| **SwanLab 日志支持** | `src/lerobot/common/wandb_utils.py` | 日志后端可选 `wandb` / `swanlab`（配置 `wandb.backend: swanlab`） |
| **Delta Action（相对动作）训练** | `src/lerobot/processor/relative_action_processor.py` | 训练时 `action -= state` 预测相对偏移，推理时加回 state；支持 `use_relative_actions` 配置 |
| **PolicyServer 推理优化** | `src/lerobot/async_inference/policy_server.py` | postprocess 由逐帧循环改为整块批量处理；推理耗时加 `cuda.synchronize` 保证计时准确 |
| **观测图片 key 过滤** | `src/lerobot/async_inference/helpers.py` | 只保留策略需要的图片 key，多余相机不再报错 |
| **数据集统计量脚本** | `src/lerobot/scripts/augment_dataset_quantile_stats.py` | 新增 `--push-to-hub` 参数，默认只改本地 |
| **数据转换工具** | `tests/my_test/dataset_create/` | mcap → LeRobot v3 格式（含里程计、多任务合并、断点续传），详见其 `readme.md` |
| **PI05 训练脚本与配置** | `tests/my_test/pi05_training/` | 完整训练流程，详见其 `README.md` |
| **SwanLab 集成测试** | `tests/swanlab/` | 后端联调测试 |
| 安装说明 | `install.md` | 依赖安装命令 |

## 环境安装（PI05 训练）

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot

# 安装 lerobot 本体及训练/推理/pi05 依赖（国内可用阿里云镜像）
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
pip install -e ".[training]" -i https://mirrors.aliyun.com/pypi/simple/
pip install -e ".[async]" -i https://mirrors.aliyun.com/pypi/simple/
pip install -e .[pi] -i https://mirrors.aliyun.com/pypi/simple/
pip install -e .[dataset] -i https://mirrors.aliyun.com/pypi/simple/

# 训练日志后端
pip install swanlab
```

## 模型训练（tests/my_test/pi05_training）

```bash
conda activate lerobot
cd tests/my_test/pi05_training

# 离线环境（模型/数据已本地缓存）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

训练前需把配置文件里的 `dataset.root` 指向本地数据集（mcap 数据需先用 `tests/my_test/dataset_create/` 工具转换）。

```bash
# 绝对动作训练（4 摄像头）
bash pi05_train_accelerate.sh

# Delta Action 训练（相对动作，推荐，收敛更快）
bash pi05_train_accelerate_deltaaction.sh

# 3 摄像头 / 其他变体
bash pi05_train_accelerate_3cam_delta.sh

# 覆盖参数
NUM_PROCESSES=4 bash pi05_train_accelerate.sh   # 4 卡训练
```

- 训练日志通过 **SwanLab** 记录（项目名 `lerobot-pi05-painting`，需设置 `SWANLAB_API_KEY`）
- 训练输出：`tests/my_test/pi05_training/outputs/train/<日期>/<时间>_<job_name>/`
- 部署用 checkpoint：`.../checkpoints/last/pretrained_model`
- Delta Action 训练前建议先跑 `recompute_relative_stats.py` 重算数据集统计量

## 模型部署（gRPC PolicyServer）

```bash
conda activate lerobot
# 远端 GPU 服务器启动 PolicyServer
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=18006 --fps=30
```

然后配置机器人端 `galbot_vla_client/vla_config.yaml`：

```yaml
policy_server_address: "<远端服务器IP>:18006"   # 指向 PolicyServer
pretrained_name_or_path: "/mnt/data1/jizheng/vla_project/lerobot/tests/my_test/pi05_training/outputs/train/<日期>/<时间>_<job>/checkpoints/last/pretrained_model"
```

启动机器人端 `python server.py` 后，即可通过 HTTP 接口下发文字指令执行 VLA 推理。
