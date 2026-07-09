#!/usr/bin/env python3
"""
阶段1: 纯拍照。屏幕显示摄像头，s 保存帧，a 退出。
完全不加载 NPU，进程退出后内核回收所有 V4L2 资源。
"""
import os
os.environ.setdefault("GST_V4L2SRC_DEFAULT_DEVICE", "/dev/video-camera0")

import cv2, time, sys, fcntl

SAVE_DIR = "/tmp/captures"


def main():
    # 清空上次的图片
    os.makedirs(SAVE_DIR, exist_ok=True)
    import glob
    for f in glob.glob(f"{SAVE_DIR}/capture_*.jpg"):
        try:
            os.remove(f)
        except:
            pass

    cap = cv2.VideoCapture("/dev/video23")
    if not cap.isOpened():
        cap = cv2.VideoCapture("/dev/video-camera0")
    if not cap.isOpened():
        cap = cv2.VideoCapture(23)
    if not cap.isOpened():
        print("CAM_FAIL")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):
        cap.read()

    cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)
    print("=" * 50)
    print("  阶段1: 实时显示 + 拍照")
    print("  s -> 保存当前帧")
    print("  a -> 结束拍照, 启动检测")
    print("=" * 50)

    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    t0, fc, fps = time.time(), 0, 0
    saved = []
    done = False

    while not done:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        key = None
        try:
            k = sys.stdin.read(1)
            key = k if k else None
        except:
            pass
        if key is None:
            try:
                k = cv2.waitKey(1) & 0xFF
                if k != 255:
                    key = chr(k)
            except:
                pass

        if key == 'a':
            done = True
        elif key == 's':
            path = f"{SAVE_DIR}/capture_{len(saved)+1:03d}.jpg"
            ok = cv2.imwrite(path, frame)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            if ok and size > 0:
                saved.append(path)
                flash = frame.copy()
                cv2.putText(flash, f"SAVED! ({len(saved)})", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("CAM", flash)
                cv2.waitKey(200)
                print(f"  [拍照] {path} {size}字节 (共 {len(saved)} 张)")
            else:
                print(f"  [错误] 保存失败! ok={ok} size={size}")

        fc += 1
        if time.time() - t0 >= 1.0:
            fps = fc / (time.time() - t0)
            fc = 0
            t0 = time.time()
        cv2.putText(frame, f"FPS:{fps:.1f} | s:save({len(saved)}) a:detect",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("CAM", frame)

    fcntl.fcntl(fd, fcntl.F_SETFL, fl)
    cap.release()
    cv2.destroyAllWindows()
    print(f"SAVED:{len(saved)}")


if __name__ == "__main__":
    main()
