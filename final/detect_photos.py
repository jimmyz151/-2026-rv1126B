#!/usr/bin/env python3
"""
阶段2: 纯 NPU 检测。读取阶段1保存的图片，批量推理，串口输出。
摄像头资源已在阶段1进程退出时被内核完全回收。
"""
import cv2, numpy as np, time, sys, os, glob

MODEL_PATH = "./best.rknn"
IMG_SIZE = 640
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"]
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
          (255, 255, 0), (0, 255, 255), (255, 0, 255)]
CONF_THRESH = 0.03
IOU_THRESH = 0.45
SAVE_DIR = "/tmp/captures"


def preprocess(img):
    h, w = img.shape[:2]
    s = min(IMG_SIZE / h, IMG_SIZE / w)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = IMG_SIZE - nw, IMG_SIZE - nh
    t, b = dh // 2, dh - dh // 2
    l, r2 = dw // 2, dw - dw // 2
    p = cv2.copyMakeBorder(r, t, b, l, r2, cv2.BORDER_CONSTANT,
                           value=(114, 114, 114))
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0,
                          0), s, l, t


def detect(rknn, frame):
    oh, ow = frame.shape[:2]
    inp, sc, pl, pt = preprocess(frame)
    out = rknn.inference(inputs=[inp])[0][0]
    boxes = out[:4, :].T
    scores = out[4:, :].T
    ids = np.argmax(scores, 1)
    confs = np.max(scores, 1)
    m = confs > CONF_THRESH
    boxes, confs, ids = boxes[m], confs[m], ids[m]
    if len(boxes) == 0:
        return []
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] -= pl
    xyxy[:, [1, 3]] -= pt
    xyxy /= sc
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, ow)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, oh)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, ow)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, oh)
    o = confs.argsort()[::-1]
    keep = []
    while o.size > 0:
        keep.append(o[0])
        if o.size == 1:
            break
        b = xyxy[o[0]]
        r = xyxy[o[1:]]
        x1 = np.maximum(b[0], r[:, 0])
        y1 = np.maximum(b[1], r[:, 1])
        x2 = np.minimum(b[2], r[:, 2])
        y2 = np.minimum(b[3], r[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        iou = inter / ((b[2] - b[0]) * (b[3] - b[1]) +
                       (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) - inter +
                       1e-6)
        o = o[1:][iou <= IOU_THRESH]
    results = []
    for i in keep:
        results.append({
            "cls": CLASS_NAMES[int(ids[i])],
            "conf": float(confs[i]),
            "box": xyxy[i].astype(int).tolist()
        })
    return results


def main():
    files = sorted(glob.glob(f"{SAVE_DIR}/capture_*.jpg"))
    if not files:
        print("NO_IMAGES")
        return

    print("=" * 50)
    print(f"  阶段2: NPU 批量检测 ({len(files)} 张)")
    print("=" * 50)

    from rknnlite.api import RKNNLite
    print("初始化 NPU...")
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    print("NPU ready")

    total_results = 0
    for i, path in enumerate(files):
        img = cv2.imread(path)
        if img is None:
            print(f"  [{i+1}/{len(files)}] {path} 读取失败")
            continue

        t1 = time.time()
        results = detect(rknn, img)
        t2 = time.time()

        print(f"\n  [{i+1}/{len(files)}] {path} ({t2-t1:.3f}s)")

        if results:
            print(f"    检测到 {len(results)} 个缺陷:")
            for r in results:
                print(f"      [{r['cls']}] conf={r['conf']:.4f} box={r['box']}")
                x1, y1, x2, y2 = r["box"]
                c = COLORS[CLASS_NAMES.index(r["cls"]) % 6]
                cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                cv2.putText(img, f"{r['cls']} {r['conf']:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            total_results += len(results)
        else:
            print(f"    未检测到缺陷")

        result_path = path.replace(".jpg", "_result.jpg")
        cv2.putText(img, f"Detect {i+1} ({t2-t1:.3f}s)", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imwrite(result_path, img)

    rknn.release()
    print(f"\n{'='*50}")
    print(f"  完成! {len(files)} 张图, 共 {total_results} 个缺陷")
    print(f"  结果图: {SAVE_DIR}/*_result.jpg")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
