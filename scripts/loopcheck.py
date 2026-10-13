"""loopcheck.py <video...> : the jump last->first vs normal frame-to-frame steps (full-res frames)."""
import cv2, numpy as np, sys, os
print(f"{'file':<52}{'last->first':>12}{'median step':>13}{'max step':>10}{'ratio':>8}  verdict")
for v in sys.argv[1:]:
    cap = cv2.VideoCapture(v); prev = first = None; steps = []
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.resize(f, (f.shape[1]//2, f.shape[0]//2), interpolation=cv2.INTER_AREA).astype(np.float32)
        if first is None: first = g
        else: steps.append(float(np.abs(g-prev).mean()))
        prev = g
    seam = float(np.abs(first-prev).mean()); med = float(np.median(steps)); mx = max(steps)
    ok = seam <= max(mx, 3*med)
    print(f"{os.path.basename(v):<52}{seam:>12.2f}{med:>13.2f}{mx:>10.2f}{seam/max(med,1e-6):>8.1f}  {'loop OK' if ok else 'JUMP'}")
