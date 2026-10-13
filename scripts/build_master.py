"""build_master.py <W> <key> <raw.png> : register the clean 4K square into the outpaint, grade the outpaint
to it, paste the exact square back with wide top/bottom ramps -> master_916.png (3072 wide) + reg_master.json"""
import cv2, numpy as np, sys, json, subprocess
W, k, raw = sys.argv[1:4]
t = cv2.imread(f'{W}/{k}/{raw}')
if t.shape[1] != 3072: t = cv2.resize(t, (3072, round(t.shape[0]*3072/t.shape[1])), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f'{W}/{k}/_raw3072.png', t)
reg = json.loads(subprocess.run([sys.executable, f'{W}/scripts/register.py', f'{W}/{k}/clean_4k.png', f'{W}/{k}/_raw3072.png'], capture_output=True, text=True).stdout)
M = np.array(reg['M'], np.float32); H, Wd = t.shape[:2]
c = cv2.imread(f'{W}/{k}/clean_4k.png')
w = cv2.warpAffine(c, M, (Wd, H), flags=cv2.INTER_AREA).astype(np.float32)
m = cv2.warpAffine(np.ones((4096, 4096), np.float32), M, (Wd, H))
inner = cv2.erode((m > 0.99).astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
tf = t.astype(np.float32); a = tf[inner].reshape(-1, 3); b = w[inner].reshape(-1, 3)
g = b.std(0)/a.std(0); o = b.mean(0)-g*a.mean(0); tf = np.clip(tf*g+o, 0, 255)
ys = np.where(m.max(1) > 0.5)[0]; y0, y1 = ys.min(), ys.max(); side = y1-y0
r = np.zeros(H, np.float32); F0, F1 = int(side*0.14), int(side*0.10)
r[y0:y1] = 1; r[y0:y0+F0] = np.linspace(0, 1, F0)**1.5; r[y1-F1:y1] = np.linspace(1, 0, F1)**1.5
mm = cv2.GaussianBlur(cv2.erode((m > 0.99).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32), (0, 0), 12)*r[:, None]
out = w*mm[..., None] + tf*(1-mm[..., None])
cv2.imwrite(f'{W}/{k}/master_916.png', out.astype(np.uint8))
reg2 = json.loads(subprocess.run([sys.executable, f'{W}/scripts/register.py', f'{W}/{k}/clean_4k.png', f'{W}/{k}/master_916.png'], capture_output=True, text=True).stdout)
json.dump(reg2, open(f'{W}/{k}/reg_master.json', 'w'))
print(k, raw, f'{Wd}x{H}', 'square rows', y0, y1, 'scale', reg2['scale'], 'ty', reg2['ty'], 'inliers', reg2['inliers'])
if len(sys.argv) > 4 and sys.argv[4] == '--seamfix':
    out = cv2.imread(f'{W}/{k}/master_916.png').astype(np.float32)
    def colmean(img): return cv2.GaussianBlur(img.mean(0, keepdims=True), (0, 0), sigmaX=50, sigmaY=0.1)[0]
    band, decay = 48, 0.15*side
    r_ = np.clip((colmean(out[y0:y0+band])+1)/(colmean(out[y0-band:y0])+1), 0.5, 1.8)
    d = (y0-1-np.arange(y0))[:, None, None]; out[:y0] = np.clip(out[:y0]*(1+(r_[None]-1)*np.exp(-d/decay)), 0, 255)
    r_ = np.clip((colmean(out[y1-band:y1])+1)/(colmean(out[y1:y1+band])+1), 0.5, 1.8)
    d = np.arange(H-y1)[:, None, None]; out[y1:] = np.clip(out[y1:]*(1+(r_[None]-1)*np.exp(-d/decay)), 0, 255)
    soft = cv2.GaussianBlur(out, (0, 0), sigmaX=0.1, sigmaY=10); yy = np.arange(H)
    wb = np.maximum(np.exp(-((yy-y0)/16.0)**2), np.exp(-((yy-y1)/16.0)**2))[:, None, None]
    out = soft*wb + out*(1-wb)
    cv2.imwrite(f'{W}/{k}/master_916.png', out.astype(np.uint8)); print(k, 'seamfix applied')
