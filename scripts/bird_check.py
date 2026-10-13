"""bird_check.py <video> : KPD QA — counts tiny dark specks (birds) in the right sky each second and measures
how fast the left smoke moves and how light changes. A changing bird count means birds merge/split/spawn."""
import cv2, numpy as np, sys
cap = cv2.VideoCapture(sys.argv[1]); i = 0; counts = []; prev = None; smoke = []; light = []
while True:
    ok, f = cap.read()
    if not ok: break
    f = cv2.resize(f, (1920, 1080))
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if i % 15 == 0:
        sky = g[0:430, 1000:1920]
        dark = ((cv2.GaussianBlur(sky, (0, 0), 6) - sky) > 18).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
        counts.append(sum(1 for k in range(1, n) if 4 <= st[k, 4] <= 120))
        sm = cv2.resize(g[0:540, 0:900], (225, 135))
        if prev is not None: smoke.append(float(np.abs(sm-prev).mean()))
        prev = sm; light.append(float(g[0:500, 1100:1920].mean()))
    i += 1
c = np.array(counts)
print(sys.argv[1].split('/')[-1], 'birds per 0.5s: min', c.min(), 'max', c.max(), 'changes', int((np.diff(c) != 0).sum()),
      '| smoke motion mean', round(np.mean(smoke), 2), '| light start/end', round(light[0], 1), round(light[-1], 1))
