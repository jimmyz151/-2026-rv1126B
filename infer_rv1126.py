"""
RV1126B 板端推理 - baseline_m 模型 (FP16)
"""
import cv2, numpy as np
from rknnlite.api import RKNNLite

IMG_SIZE = 640
CLASS_NAMES = ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
RKNN_PATH = "./best.rknn"
CONF_THRESH = 0.03; IOU_THRESH = 0.45

def preprocess(img_bgr):
    h,w = img_bgr.shape[:2]; s = min(IMG_SIZE/h, IMG_SIZE/w)
    nw,nh = int(w*s), int(h*s)
    r = cv2.resize(img_bgr, (nw,nh), interpolation=cv2.INTER_LINEAR)
    dw,dh = IMG_SIZE-nw, IMG_SIZE-nh
    t,b = dh//2, dh-dh//2; l,r2 = dw//2, dw-dw//2
    p = cv2.copyMakeBorder(r,t,b,l,r2,cv2.BORDER_CONSTANT,value=(114,114,114))
    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2,0,1).astype(np.float32) / 255.0
    return np.expand_dims(chw,0), s, l, t

def postprocess(output0, scale, pad_l, pad_t, oh, ow):
    # output0: (1, 10, 8400) = Concat(bbox[0:4], sigmoid(cls[4:10]))
    boxes = output0[0, :4, :].T   # (8400, 4)
    scores = output0[0, 4:, :].T  # (8400, 6), 已经是sigmoid后的值
    ids = np.argmax(scores,1); confs = np.max(scores,1)
    m = confs > CONF_THRESH
    boxes, confs, ids = boxes[m], confs[m], ids[m]
    if len(boxes) == 0: return []
    xyxy = np.zeros_like(boxes)
    xyxy[:,0]=boxes[:,0]-boxes[:,2]/2; xyxy[:,1]=boxes[:,1]-boxes[:,3]/2
    xyxy[:,2]=boxes[:,0]+boxes[:,2]/2; xyxy[:,3]=boxes[:,1]+boxes[:,3]/2
    xyxy[:,[0,2]]-=pad_l; xyxy[:,[1,3]]-=pad_t; xyxy/=scale
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
    for k in keep:
        results.append({"bbox":xyxy[k].astype(int).tolist(),"score":float(confs[k]),
                        "class":CLASS_NAMES[int(ids[k])]})
    return results

def main():
    import sys
    if len(sys.argv)<2: print("python infer_rv1126.py <img>"); return
    rknn = RKNNLite(); rknn.load_rknn(RKNN_PATH); rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    img = cv2.imread(sys.argv[1])
    if img is None: print("read fail"); rknn.release(); return
    oh,ow = img.shape[:2]
    inp,sc,pl,pt = preprocess(img)
    out = rknn.inference(inputs=[inp])[0]
    print(f"output: {out.shape} [{out.min():.2f},{out.max():.2f}]")
    results = postprocess(out, sc, pl, pt, oh, ow)
    print(f"detected: {len(results)}")
    colors=[(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,255,255),(255,0,255)]
    for r in results:
        print(f"  {r['class']}: {r['score']:.4f} @ {r['bbox']}")
        x1,y1,x2,y2=r["bbox"]; c=colors[CLASS_NAMES.index(r["class"])%6]
        cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
        cv2.putText(img,f"{r['class']} {r['score']:.2f}",(x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1)
    cv2.imwrite("./result.jpg",img)
    rknn.release()

if __name__=="__main__": main()
