"""
baseline_m ONNX -> RKNN 转换 (FP16)
运行环境: 虚拟机
命令: python3 convert_rknn.py
"""
from rknn.api import RKNN

ONNX_PATH = "./best.onnx"
RKNN_PATH = "./best.rknn"
RKNN_PLATFORM = "rv1126b"


def main():
    rknn = RKNN(verbose=True)

    print("\n[1/4] 配置 RKNN (FP16)...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],  # 归一化由Python端/255处理
        quant_img_RGB2BGR=False,
        target_platform=RKNN_PLATFORM,
    )

    print("\n[2/4] 加载 ONNX...")
    ret = rknn.load_onnx(model=ONNX_PATH)
    if ret != 0:
        print(f"加载失败! ret={ret}")
        return

    print("\n[3/4] 构建 RKNN (FP16)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"构建失败! ret={ret}")
        return

    print("\n[4/4] 导出 RKNN...")
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        print(f"导出失败! ret={ret}")
        return
    print(f"已导出: {RKNN_PATH}")

    rknn.release()
    print("完成!")


if __name__ == "__main__":
    main()
