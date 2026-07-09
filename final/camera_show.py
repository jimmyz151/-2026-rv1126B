#!/usr/bin/env python3
"""
纯摄像头推屏程序。被 detect_trigger.py 调用。
屏幕显示实时画面, 同时每帧存到 /tmp/last_frame.jpg
用法: python camera_show.py
"""
import os
os.environ.setdefault("GST_V4L2SRC_DEFAULT_DEVICE", "/dev/video-camera0")
import cv2, time, signal, sys

running = True

def on_signal(sig, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, on_signal)

cap = cv2.VideoCapture("/dev/video23")
if not cap.isOpened(): cap = cv2.VideoCapture("/dev/video-camera0")
if not cap.isOpened(): cap = cv2.VideoCapture(23)
if not cap.isOpened():
    print("CAM_FAIL"); sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(5): cap.read()

cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)
print("CAM_OK")
sys.stdout.flush()

t0, fc, fps = time.time(), 0, 0
last_save = 0

while running:
    ret, frame = cap.read()
    if not ret: time.sleep(0.01); continue

    fc += 1
    if time.time() - t0 >= 1.0:
        fps = fc / (time.time() - t0); fc = 0; t0 = time.time()

    cv2.putText(frame, f"FPS:{fps:.1f}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.imshow("CAM", frame)
    cv2.waitKey(1)

    # 每 0.5 秒存一帧到 /tmp
    if time.time() - last_save > 0.5:
        cv2.imwrite("/tmp/last_frame.jpg", frame)
        last_save = time.time()

cap.release()
cv2.destroyAllWindows()
