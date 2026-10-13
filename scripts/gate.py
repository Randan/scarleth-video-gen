"""gate.py <ref> <cand> [out.jpg crop...] -> alignment + edge overlap (+ side-by-side crops)"""
import cv2, numpy as np, sys
ref = cv2.imread(sys.argv[1]); cand = cv2.imread(sys.argv[2])
S = 1024
a = cv2.cvtColor(cv2.resize(ref, (S, int(S*ref.shape[0]/ref.shape[1]))), cv2.COLOR_BGR2GRAY).astype(np.float32)
b = cv2.cvtColor(cv2.resize(cand, (a.shape[1], a.shape[0])), cv2.COLOR_BGR2GRAY).astype(np.float32)
(dx, dy), r = cv2.phaseCorrelate(a, b)
ga = cv2.Canny(a.astype(np.uint8), 60, 150); gb = cv2.Canny(b.astype(np.uint8), 60, 150)
ov = float((cv2.dilate(ga, np.ones((5, 5))) > 0)[gb > 0].mean())
print(f'shift {dx:.1f},{dy:.1f}  resp {r:.3f}  edge-overlap {ov:.3f}')
if len(sys.argv) > 4:
    crops = []
    for c in sys.argv[4:]:
        x0, y0, x1, y1 = map(float, c.split(','))
        def cr(im):
            H, Wd = im.shape[:2]; p = im[int(y0*H):int(y1*H), int(x0*Wd):int(x1*Wd)]
            return cv2.resize(p, (700, max(1, int(700*(y1-y0)/(x1-x0)))))
        crops.append(np.hstack([cr(ref), cr(cand)]))
    cv2.imwrite(sys.argv[3], np.vstack(crops), [cv2.IMWRITE_JPEG_QUALITY, 88])
