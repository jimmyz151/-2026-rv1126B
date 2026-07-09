#!/bin/bash
# 两阶段独立进程: 拍照进程退出后内核回收 V4L2 资源, 再启动纯 NPU 检测
# 这样 NPU 和摄像头不会同时存在于同一进程空间

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/elf/demo/elf-env/bin/python
export DISPLAY=:0

echo "=== 阶段1: 拍照 ==="
$PYTHON "$SCRIPT_DIR/capture_photos.py"
SAVE_EXIT=$?

if [ $SAVE_EXIT -ne 0 ]; then
    echo "拍照出错, 退出"
    exit 1
fi

# 确保摄像头进程完全退出, 内核回收资源
sleep 1

echo ""
echo "=== 阶段2: NPU 检测 ==="
$PYTHON "$SCRIPT_DIR/detect_photos.py"

echo ""
echo "=== 全部完成 ==="
