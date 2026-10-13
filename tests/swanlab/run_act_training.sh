#!/bin/bash
# 使用 SwanLab 训练 ACT 模型的快速启动脚本

set -e

echo "============================================================"
echo "使用 SwanLab 训练 ACT 模型"
echo "============================================================"
echo ""

# 配置
API_KEY="Zu2D1H0CeWLxCZjg7H"
PROJECT="lerobot-act-test"
EPOCHS=10
BATCH_SIZE=32
DATASET="lerobot/svla_so101_pickplace"
DEVICE="cuda"

echo "配置信息:"
echo "  API Key: ${API_KEY:0:8}...${API_KEY: -8}"
echo "  项目：${PROJECT}"
echo "  数据集：${DATASET}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  设备：${DEVICE}"
echo ""

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import torch" 2>/dev/null || {
    echo "警告：torch 未安装"
    read -p "是否安装 torch? (y/n): " install
    if [ "$install" = "y" ]; then
        pip install torch
    fi
}

python3 -c "import swanlab" 2>/dev/null || {
    echo "警告：swanlab 未安装"
    read -p "是否安装 swanlab? (y/n): " install
    if [ "$install" = "y" ]; then
        pip install swanlab
    fi
}

echo ""
echo "============================================================"
echo "开始训练"
echo "============================================================"
echo ""

# 运行训练
python3 tests/swanlab/train_act_with_swanlab.py \
    --api-key "$API_KEY" \
    --project "$PROJECT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --dataset "$DATASET" \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo "训练完成!"
echo "============================================================"
echo ""
echo "查看日志:"
echo "  本地：swanlab watch outputs/act_swanlab/swanlog"
echo "  云端：https://swanlab.cn"
echo ""
