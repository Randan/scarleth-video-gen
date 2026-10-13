"""grade_stills.py <W>

Gemini re-renders drift in colour. This pulls every 4K still back onto its reference with one
per-channel gain/offset (mean/std match):
  sq_text_4k (kpd, bts, slv)  -> orig.png
  clean_4k   (kpd, bts)       -> sq_text_4k, measured only on pixels far from the lettering
  clean_4k   (slv)            -> clean_src.png
Run after the fidelity and lettering-removal Gemini passes, before build_master.py.
"""
import cv2, numpy as np, sys

W = sys.argv[1]

def grade(src, ref, mask=None):
    a = cv2.resize(src, (1024, 1024)).astype(np.float64); b = cv2.resize(ref, (1024, 1024)).astype(np.float64)
    if mask is not None: a, b = a[mask], b[mask]
    a = a.reshape(-1, 3); b = b.reshape(-1, 3)
    g = b.std(0)/a.std(0); o = b.mean(0)-g*a.mean(0)
    return np.clip(src.astype(np.float32)*g.astype(np.float32)+o.astype(np.float32), 0, 255).astype(np.uint8)

for k in ['kpd', 'bts', 'slv']:
    cv2.imwrite(f'{W}/{k}/sq_text_4k.png', grade(cv2.imread(f'{W}/{k}/sq_text_4k.png'), cv2.imread(f'{W}/{k}/orig.png')))
for k in ['kpd', 'bts']:
    t = cv2.imread(f'{W}/{k}/sq_text_4k.png'); c = cv2.imread(f'{W}/{k}/clean_4k.png')
    d = np.abs(cv2.GaussianBlur(cv2.resize(t, (1024, 1024)).astype(np.float64), (0, 0), 4)
               - cv2.GaussianBlur(cv2.resize(c, (1024, 1024)).astype(np.float64), (0, 0), 4)).mean(2)
    cv2.imwrite(f'{W}/{k}/clean_4k.png', grade(c, t, d < np.percentile(d, 80)))
cv2.imwrite(f'{W}/slv/clean_4k.png', grade(cv2.imread(f'{W}/slv/clean_4k.png'), cv2.imread(f'{W}/slv/clean_src.png')))
print('graded')
