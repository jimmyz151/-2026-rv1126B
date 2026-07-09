"""
串口输入图片名 → 检测 → 屏幕显示结果
用法: python screen_detect.py
"""
import cv2, numpy as np, os
os.environ['DISPLAY'] = ':0'
from rknnlite.api import RKNNLite

MODEL_PATH = "./best.rknn"
IMG_SIZE = 640
CLASS_NAMES = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
COLORS = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255),(255,0,255)]
CONF_THRESH = 0.03; IOU_THRESH = 0.45

def preprocess(img):
    h,w = img.shape[:2]; s = min(IMG_SIZE/h, IMG_SIZE/w)
    nw,nh = int(w*s), int(h*s)
    r = cv2.resize(img, (nw,nh), interpolation=cv2.INTER_LINEAR)
    dw,dh = IMG_SIZE-nw, IMG_SIZE-nh
    t,b = dh//2, dh-dh//2; l,r2 = dw//2, dw-dw//2
    p = cv2.copyMakeBorder(r,t,b,l,r2,cv2.BORDER_CONSTANT,value=(114,114,114))
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb.transpose(2,0,1).astype(np.float32)/255.0, 0), s, l, t

def detect(rknn, frame):
    oh, ow = frame.shape[:2]
    inp, sc, pl, pt = preprocess(frame)
    out = rknn.inference(inputs=[inp])[0][0]  # (10, 8400)
    boxes = out[:4,:].T; scores = out[4:,:].T
    ids = np.argmax(scores,1); confs = np.max(scores,1)
    m = confs > CONF_THRESH
    boxes, confs, ids = boxes[m], confs[m], ids[m]
    if len(boxes) == 0: return []
    xyxy = np.zeros_like(boxes)
    xyxy[:,0]=boxes[:,0]-boxes[:,2]/2; xyxy[:,1]=boxes[:,1]-boxes[:,3]/2
    xyxy[:,2]=boxes[:,0]+boxes[:,2]/2; xyxy[:,3]=boxes[:,1]+boxes[:,3]/2
    xyxy[:,[0,2]]-=pl; xyxy[:,[1,3]]-=pt; xyxy/=sc
    xyxy[:,0]=np.clip(xyxy[:,0],0,ow); xyxy[:,1]=np.clip(xyxy[:,1],0,oh)
    xyxy[:,2]=np.clip(xyxy[:,2],0,ow); xyxy[:,3]=np.clip(xyxy[:,3],0,oh)
    o = confs.argsort()[::-1]; keep = []
    while o.size > 0:
        keep.append(o[0])
        if o.size == 1: break
        b = xyxy[o[0]]; r = xyxy[o[1:]]
        x1=np.maximum(b[0],r[:,0]); y1=np.maximum(b[1],r[:,1])
        x2=np.minimum(b[2],r[:,2]); y2=np.minimum(b[3],r[:,3])
        inter = np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
        iou = inter/((b[2]-b[0])*(b[3]-b[1])+(r[:,2]-r[:,0])*(r[:,3]-r[:,1])-inter+1e-6)
        o = o[1:][iou<=IOU_THRESH]
    results = []
    for i in keep:
        results.append({"cls":CLASS_NAMES[int(ids[i])],"conf":float(confs[i]),
                        "box":xyxy[i].astype(int).tolist()})
    return results

def draw(img, results):
    for r in results:
        x1,y1,x2,y2 = r["box"]
        c = COLORS[CLASS_NAMES.index(r["cls"])%6]
        cv2.rectangle(img, (x1,y1), (x2,y2), c, 2)
        cv2.putText(img, f"{r['cls']} {r['conf']:.2f}", (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return img

def main():
    img_dir = "/run/media/sda1"  # U盘图片目录

    print("init NPU...")
    rknn = RKNNLite()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    print("ready")

    # 全屏窗口
    cv2.namedWindow("DETECT", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("DETECT", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        print("\n输入图片名 (如 crazing_2.jpg, q退出):")
        name = input("> ").strip()
        if name.lower() == 'q':
            break

        path = os.path.join(img_dir, name)
        if not os.path.exists(path):
            print(f"文件不存在: {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            print("读取失败")
            continue

        results = detect(rknn, img)
        print(f"检测到 {len(results)} 个:")
        for r in results:
            print(f"  {r['cls']}: {r['conf']:.4f} @ {r['box']}")

        display = draw(img.copy(), results)
        cv2.imshow("DETECT", display)
        cv2.waitKey(1)  # 刷新窗口

    cv2.destroyAllWindows()
    rknn.release()

if __name__ == "__main__":
    main()
