"""analyze.py <W> <key> <video> : camera drift vs frame 0 (streamed), activity heat + contact sheet."""
import cv2, numpy as np, sys
W, k, vid = sys.argv[1:4]
cap = cv2.VideoCapture(f'{W}/{k}/{vid}'); i = 0; g0 = None; sh = []; acc = acc2 = None; n = 0; keep = {}
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(f, (540, 960)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g0 is None: g0 = g
    if i % 15 == 0:
        (dx, dy), r = cv2.phaseCorrelate(g0, g); sh.append((dx*2, dy*2, r))
    if i % 90 == 0: keep[i] = cv2.resize(f, (270, 480))
    s = cv2.GaussianBlur(cv2.resize(g, (270, 480)), (0, 0), 1.5).astype(np.float64)
    acc = s if acc is None else acc+s; acc2 = s*s if acc2 is None else acc2+s*s; n += 1; i += 1
act = np.sqrt(np.maximum(acc2/n-(acc/n)**2, 0))
print(k, 'frames', i, 'max drift px', round(max(abs(a) for a, b, c in sh), 2), round(max(abs(b) for a, b, c in sh), 2), 'min resp', round(min(c for a, b, c in sh), 3))
heat = cv2.applyColorMap(np.clip(act*12, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite(f'{W}/qa/{k}_analysis.jpg', np.hstack(list(keep.values())+[cv2.addWeighted(keep[0], 0.5, heat, 0.5, 0)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
