"""text_layer.py <W> <key> : lettering RGBA on the 4096 square = pixels where the cover differs from the clean art."""
import cv2, numpy as np, sys
W, k = sys.argv[1:3]
t = cv2.imread(f'{W}/{k}/sq_text_4k.png').astype(np.float32); c = cv2.imread(f'{W}/{k}/clean_4k.png').astype(np.float32)
d = np.abs(cv2.GaussianBlur(t, (0, 0), 2)-cv2.GaussianBlur(c, (0, 0), 2)).max(2)
mk = cv2.morphologyEx((d > 28).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(mk); keep = np.zeros_like(mk)
for i in range(1, n):
    if st[i, 4] > 1500: keep[lab == i] = 1
keep = cv2.morphologyEx(cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
a = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 4)
cv2.imwrite(f'{W}/{k}/text_layer.png', np.dstack([t, a*255]).astype(np.uint8))
blk = cv2.dilate((a > 0.15).astype(np.uint8), np.ones((61, 121), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(blk)
print(k, 'blocks', [tuple(int(v) for v in s) for s in st[1:] if s[4] > 20000])
