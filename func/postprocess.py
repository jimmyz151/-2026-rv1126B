"""
YOLOv8 后处理 — 单输出格式 (1, 4+num_cls, num_boxes)
兼容 baseline_m 模型, 用于 rknnPoolExecutor
"""
import cv2
import numpy as np
from typing import Tuple, Optional

def create_postprocessor(
    obj_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    img_size: int = 640,
    classes: Tuple[str, ...] = (),
):
    _obj_thresh = obj_thresh
    _nms_thresh = nms_thresh
    _img_size = img_size
    _classes = classes
    COLORS = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255),(255,0,255)]

    def preprocess(img_bgr):
        h, w = img_bgr.shape[:2]
        s = min(_img_size / h, _img_size / w)
        nw, nh = int(w * s), int(h * s)
        r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = _img_size - nw, _img_size - nh
        t, b = dh // 2, dh - dh // 2
        l, r2 = dw // 2, dw - dw // 2
        p = cv2.copyMakeBorder(r, t, b, l, r2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(chw, 0), s, l, t

    def nms(boxes, scores, th):
        o = scores.argsort()[::-1]; k = []
        while o.size > 0:
            k.append(o[0])
            if o.size == 1: break
            b = boxes[o[0]]; r = boxes[o[1:]]
            x1 = np.maximum(b[0], r[:, 0]); y1 = np.maximum(b[1], r[:, 1])
            x2 = np.minimum(b[2], r[:, 2]); y2 = np.minimum(b[3], r[:, 3])
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            iou = inter / ((b[2] - b[0]) * (b[3] - b[1]) + (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) - inter + 1e-6)
            o = o[1:][iou <= th]
        return k

    def myFunc(rknn_lite, IMG: np.ndarray) -> np.ndarray:
        oh, ow = IMG.shape[:2]
        inp, sc, pl, pt = preprocess(IMG)
        out = rknn_lite.inference(inputs=[inp])[0][0]  # (1, 4+cls, 8400)

        boxes = out[:4, :].T; scores = out[4:, :].T
        ids = np.argmax(scores, 1); confs = np.max(scores, 1)
        m = confs > _obj_thresh
        boxes, confs, ids = boxes[m], confs[m], ids[m]

        if len(boxes) > 0:
            xyxy = np.zeros_like(boxes)
            xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            xyxy[:, [0, 2]] -= pl; xyxy[:, [1, 3]] -= pt; xyxy /= sc
            xyxy[:, 0] = np.clip(xyxy[:, 0], 0, ow); xyxy[:, 1] = np.clip(xyxy[:, 1], 0, oh)
            xyxy[:, 2] = np.clip(xyxy[:, 2], 0, ow); xyxy[:, 3] = np.clip(xyxy[:, 3], 0, oh)

            keep = nms(xyxy, confs, _nms_thresh)
            for k in keep:
                x1, y1, x2, y2 = xyxy[k].astype(int)
                cls_id = int(ids[k])
                c = COLORS[cls_id % 6]
                cv2.rectangle(IMG, (x1, y1), (x2, y2), c, 2)
                label = _classes[cls_id] if cls_id < len(_classes) else f"cls_{cls_id}"
                cv2.putText(IMG, f"{label} {confs[k]:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        return IMG

    return myFunc
