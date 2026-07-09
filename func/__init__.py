# func — YOLOv8 后处理函数（板端 NPU 推理用）
from .postprocess import create_postprocessor

__all__ = ["create_postprocessor"]
