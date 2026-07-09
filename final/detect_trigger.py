#!/usr/bin/env python3
"""
主控: 摄像头推屏(camera_show.py) + 串口按键触发 NPU 检测
用法: python detect_trigger.py
"""
import os, sys, time, subprocess, signal, fcntl
import cv2, numpy as np

MODEL_PATH = "./best.rknn"
IMG_SIZE = 640
CLASS_NAMES = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
COLORS = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255),(255,0,255)]
CONF_THRESH = 0.03; IOU_THRESH = 0.45

CAMERA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_show.py")

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

def start_camera():
    proc = subprocess.Popen(
        [sys.executable, CAMERA_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for _ in range(50):
        line = proc.stdout.readline()
        if line:
            line = line.decode().strip()
            if line == "CAM_OK":
                print("[主控] 摄像头已启动")
                return proc
            if line == "CAM_FAIL":
                print("[主控] 摄像头启动失败")
                proc.terminate(); proc.wait()
                return None
        time.sleep(0.1)
    print("[主控] 摄像头启动超时")
    proc.terminate(); proc.wait()
    return None

def main():
    from rknnlite.api import RKNNLite

    # 1. 先初始化 NPU（摄像头还没开）
    print("[主控] 初始化 NPU...")
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    print("[主控] NPU ready")

    # 2. 再启动摄像头
    cam_proc = start_camera()
    if cam_proc is None:
        print("[主控] 无法启动摄像头, 退出")
        rknn.release(); return

    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    infer_count = 0
    print("=" * 50)
    print("  s/Enter -> NPU检测（不关摄像头）")
    print("  q       -> 退出")
    print("=" * 50)

    try:
        while True:
            key = None
            try: k = sys.stdin.read(1); key = k if k else None
            except: pass
            if key is None:
                try:
                    k = cv2.waitKey(1) & 0xFF
                    if k != 255: key = chr(k)
                except: pass
            if key == 'q': break

            if key and key in 's\r\n':
                print(f"\n>>> 检测 #{infer_count+1} ...")
                img = cv2.imread("/tmp/last_frame.jpg")
                if img is None:
                    print("  !!! 读取 /tmp/last_frame.jpg 失败")
                    continue

                t1 = time.time()
                results = detect(rknn, img)
                t2 = time.time()

                # 画框存图
                for r in results:
                    x1,y1,x2,y2 = r["box"]
                    c = COLORS[CLASS_NAMES.index(r["cls"])%6]
                    cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
                    cv2.putText(img,f"{r['cls']} {r['conf']:.2f}",(x1,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,c,2)
                cv2.putText(img,f"Detect #{infer_count+1} ({t2-t1:.3f}s)",
                            (4,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
                cv2.imwrite(f"/tmp/result_{infer_count+1}.jpg", img)

                if results:
                    print(f"  检测到 {len(results)} 个缺陷 ({t2-t1:.3f}s):")
                    for r in results:
                        print(f"    [{r['cls']}] conf={r['conf']:.4f} box={r['box']}")
                else:
                    print(f"  未检测到缺陷 ({t2-t1:.3f}s)")
                print(f"  结果图: /tmp/result_{infer_count+1}.jpg")
                print("---")
                infer_count += 1

    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, fl)
        cam_proc.terminate()
        try: cam_proc.wait(timeout=3)
        except: cam_proc.kill(); cam_proc.wait()
        rknn.release()
        print(f"\n共检测 {infer_count} 次, 退出")

if __name__ == "__main__":
    main()
