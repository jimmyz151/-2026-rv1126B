import cv2, numpy as np
from rknnlite.api import RKNNLite

rknn = RKNNLite()
rknn.load_rknn('./best.rknn')
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

img = cv2.imread('/run/media/sda1/crazing_2.jpg')
h,w = img.shape[:2]
s = min(640/h, 640/w)
nw,nh = int(w*s), int(h*s)
r = cv2.resize(img, (nw,nh))
dw,dh = 640-nw, 640-nh
t,b = dh//2, dh-dh//2
l,r2 = dw//2, dw-dw//2
p = cv2.copyMakeBorder(r,t,b,l,r2,cv2.BORDER_CONSTANT,value=(114,114,114))
rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
inp = np.expand_dims(rgb.transpose(2,0,1).astype(np.float32)/255.0, 0)

out = rknn.inference(inputs=[inp])[0][0]
cls_scores = out[4:, :]
print('cls range:', cls_scores.min(), cls_scores.max())
for ci in range(6):
    print(f'class {ci}: min={cls_scores[ci].min():.4f} max={cls_scores[ci].max():.4f}')
max_per = cls_scores.max(axis=0)
print('top10 scores:', sorted(max_per.tolist(), reverse=True)[:10])
rknn.release()
