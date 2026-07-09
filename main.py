#!/usr/bin/env python3
"""
屏幕实时显示摄像头画面，按 s 键: 关摄像头→NPU检测→重开摄像头
"""
import os
os.environ.setdefault("GST_V4L2SRC_DEFAULT_DEVICE", "/dev/video-camera0")

import cv2, numpy as np, time, sys, fcntl

MODEL_PATH = "./best.rknn"
IMG_SIZE = 640
CLASS_NAMES = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
COLORS = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255),(255,0,255)]
CONF_THRESH = 0.03
IOU_THRESH = 0.45

def open_cam():
    cap = cv2.VideoCapture("/dev/video23")
    if not cap.isOpened(): cap = cv2.VideoCapture("/dev/video-camera0")
    if not cap.isOpened(): cap = cv2.VideoCapture(23)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5): cap.read()
    return cap

def preprocess(img):
    h,w = img.shape[:2]; s = min(IMG_SIZE/h, IMG_SIZE/w)
    nw,nh = int(w*s), int(h*s)
    r = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
    dw,dh = IMG_SIZE-nw, IMG_SIZE-nh
    t,b = dh//2, dh-dh//2; l,r2 = dw//2, dw-dw//2
    p = cv2.copyMakeBorder(r,t,b,l,r2,cv2.BORDER_CONSTANT,value=(114,114,114))
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.transpose(2,0,1).astype(np.float32)/255.0,0), s, l, t

def detect(rknn, frame):
    oh,ow = frame.shape[:2]
    inp,sc,pl,pt = preprocess(frame)
    out = rknn.inference(inputs=[inp])[0][0]
    boxes = out[:4,:].T; scores = out[4:,:].T
    ids = np.argmax(scores,1); confs = np.max(scores,1)
    m = confs > CONF_THRESH
    boxes,confs,ids = boxes[m],confs[m],ids[m]
    if len(boxes)==0: return []
    xyxy = np.zeros_like(boxes)
    xyxy[:,0]=boxes[:,0]-boxes[:,2]/2; xyxy[:,1]=boxes[:,1]-boxes[:,3]/2
    xyxy[:,2]=boxes[:,0]+boxes[:,2]/2; xyxy[:,3]=boxes[:,1]+boxes[:,3]/2
    xyxy[:,[0,2]]-=pl; xyxy[:,[1,3]]-=pt; xyxy/=sc
    xyxy[:,0]=np.clip(xyxy[:,0],0,ow); xyxy[:,1]=np.clip(xyxy[:,1],0,oh)
    xyxy[:,2]=np.clip(xyxy[:,2],0,ow); xyxy[:,3]=np.clip(xyxy[:,3],0,oh)
    o = confs.argsort()[::-1]; keep = []
    while o.size>0:
        keep.append(o[0])
        if o.size==1: break
        b=xyxy[o[0]]; r=xyxy[o[1:]]
        x1=np.maximum(b[0],r[:,0]); y1=np.maximum(b[1],r[:,1])
        x2=np.minimum(b[2],r[:,2]); y2=np.minimum(b[3],r[:,3])
        inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
        iou=inter/((b[2]-b[0])*(b[3]-b[1])+(r[:,2]-r[:,0])*(r[:,3]-r[:,1])-inter+1e-6)
        o=o[1:][iou<=IOU_THRESH]
    results=[]
    for i in keep:
        results.append({"cls":CLASS_NAMES[int(ids[i])],"conf":float(confs[i]),
                        "box":xyxy[i].astype(int).tolist()})
    return results

def main():
    from rknnlite.api import RKNNLite
    display = "--display" in sys.argv

    print("init NPU...")
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    print("NPU ready")

    cap = open_cam()
    if cap is None: print("cam fail"); rknn.release(); return
    print("camera ready")

    if display:
        cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)
        print("display ready")

    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    t0, fc, fps = time.time(), 0, 0
    infer_count = 0

    print("=" * 50)
    print("  s/Enter -> 关摄像头 -> NPU检测 -> 重开摄像头")
    print("  q       -> 退出")
    print("=" * 50)

    try:
        while True:
            # 按键检测
            key = None
            try: k = sys.stdin.read(1); key = k if k else None
            except: pass
            if display and key is None:
                k = cv2.waitKey(1) & 0xFF
                if k != 255: key = chr(k)
            if key == 'q': break

            if key and key in 's\r\n':
                print(f"\n>>> 检测 #{infer_count+1}: 画面定格, NPU推理中...")
                snap = last_frame.copy()
                t1 = time.time()
                results = detect(rknn, snap)
                t2 = time.time()

                # 画框
                result_img = snap.copy()
                for r in results:
                    x1,y1,x2,y2 = r["box"]
                    c = COLORS[CLASS_NAMES.index(r["cls"])%6]
                    cv2.rectangle(result_img,(x1,y1),(x2,y2),c,2)
                    cv2.putText(result_img,f"{r['cls']} {r['conf']:.2f}",(x1,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,c,2)
                cv2.putText(result_img,f"Detect #{infer_count+1} ({t2-t1:.3f}s)",
                            (4,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)

                # 显示结果 2 秒
                if display:
                    show_until = time.time() + 2.0
                    while time.time() < show_until:
                        cv2.imshow("CAM", result_img)
                        cv2.waitKey(1)

                # 串口输出
                if results:
                    print(f"    检测到 {len(results)} 个缺陷 ({t2-t1:.3f}s):")
                    for r in results:
                        print(f"      [{r['cls']}] conf={r['conf']:.4f} box={r['box']}")
                else:
                    print(f"    未检测到缺陷 ({t2-t1:.3f}s)")

                infer_count += 1
                print("---")

            # 持续读摄像头推屏
            ret, frame = cap.read()
            if not ret: time.sleep(0.01); continue
            last_frame = frame.copy()

            fc += 1
            if time.time() - t0 >= 1.0:
                fps = fc / (time.time() - t0); fc = 0; t0 = time.time()

            cv2.putText(frame,f"FPS:{fps:.1f} | s:detect q:quit",(4,14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
            if display:
                cv2.imshow("CAM", frame)

    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, fl)
        cap.release()
        if display: cv2.destroyAllWindows()
        print(f"\n共检测 {infer_count} 次, 退出")

if __name__ == "__main__":
    main()
