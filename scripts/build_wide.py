"""build_wide.py <W> <key> <raw.png>

16:9 stage master (3840x2160, textless): registers the clean 4K square into the outpaint, grades the
outpaint to it on their overlap, and pastes the exact square back with wide LEFT/RIGHT feathers, so the
original art stays crisp and the join sits inside the painting. -> <key>/wide_master.png + wide_reg.json
"""
import cv2, numpy as np, sys, json, subprocess, os

W, k, raw = sys.argv[1:4]
OW, OH = 3840, 2160
t = cv2.resize(cv2.imread(f'{W}/{k}/{raw}'), (OW, OH), interpolation=cv2.INTER_AREA)
tmp = f'{W}/{k}/_wide_raw_3840.png'; cv2.imwrite(tmp, t)
reg = json.loads(subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'register.py'),
                                 f'{W}/{k}/clean_4k.png', tmp], capture_output=True, text=True).stdout)
M = np.array(reg['M'], np.float32)
c = cv2.imread(f'{W}/{k}/clean_4k.png')
w = cv2.warpAffine(c, M, (OW, OH), flags=cv2.INTER_AREA).astype(np.float32)
m = cv2.warpAffine(np.ones((4096, 4096), np.float32), M, (OW, OH))
inner = cv2.erode((m > 0.99).astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
tf = t.astype(np.float32); a = tf[inner].reshape(-1, 3); b = w[inner].reshape(-1, 3)
g = b.std(0)/a.std(0); o = b.mean(0)-g*a.mean(0); tf = np.clip(tf*g+o, 0, 255)
xs = np.where(m.max(0) > 0.5)[0]; x0, x1 = xs.min(), xs.max(); side = x1-x0
r = np.zeros(OW, np.float32); F = int(side*0.12)
r[x0:x1] = 1; r[x0:x0+F] = np.linspace(0, 1, F)**1.5; r[x1-F:x1] = np.linspace(1, 0, F)**1.5
mm = cv2.GaussianBlur(cv2.erode((m > 0.99).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32), (0, 0), 12)*r[None, :]
out = w*mm[..., None] + tf*(1-mm[..., None])
cv2.imwrite(f'{W}/{k}/wide_master.png', out.astype(np.uint8)); os.remove(tmp)
reg2 = json.loads(subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'register.py'),
                                  f'{W}/{k}/clean_4k.png', f'{W}/{k}/wide_master.png'], capture_output=True, text=True).stdout)
json.dump(reg2, open(f'{W}/{k}/wide_reg.json', 'w'))
print(k, raw, 'square cols', x0, x1, 'scale', reg2['scale'], 'inliers', reg2['inliers'])
