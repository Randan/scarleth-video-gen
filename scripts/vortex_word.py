"""vortex_word.py <W>

Vortex only: lifts the glowing word "VORTEX" off the cover as a clean RGBA (vtx/vortex_word_rgba.png)
for the 3:4 layout, where it sits on a different background. Cutting it with the generic text matte
dragged a rectangle of the cover's dark background along (it read as a burnt-in patch on the new
background). Here the alpha is solved against the clean art (screen-style unblend), kept full only on
the letter strokes, and limited to a soft glow envelope around them so stray shard edges drop out.
"""
import cv2, numpy as np, sys

W = sys.argv[1]
C = cv2.imread(f'{W}/vtx/sq_text_4k.png').astype(np.float32); B = cv2.imread(f'{W}/vtx/clean_4k.png').astype(np.float32)
x, y, w, h = 2499, 3472, 1328, 379                     # word box on the 4096 cover, with margin
c = C[y:y+h, x:x+w]; b = cv2.GaussianBlur(B[y:y+h, x:x+w], (0, 0), 2)
a = np.clip((c-b)/(255-b+1e-3), 0, 1).max(2)
body = cv2.morphologyEx((a > 0.85).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(body)
keep = np.zeros_like(body)
for i in range(1, n):
    if st[i, 4] > 400 and st[i, 3] > 80: keep[lab == i] = 1   # letter strokes, not thin shard slivers
env = np.clip(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 16)*3.0, 0, 1)
ag = cv2.medianBlur((a*255).astype(np.uint8), 11).astype(np.float32)/255
af = np.where(cv2.dilate(keep, np.ones((3, 3), np.uint8)) > 0, a, np.minimum(ag, env))
af = cv2.GaussianBlur(af, (0, 0), 0.8); af[af < 0.04] = 0
F = np.clip((c-(1-af[..., None])*b)/np.maximum(af[..., None], 0.05), 0, 255)
F = np.where(af[..., None] < 0.6, np.clip(cv2.GaussianBlur(F, (0, 0), 3), 0, 255), F)
cv2.imwrite(f'{W}/vtx/vortex_word_rgba.png', np.dstack([F, af*255]).astype(np.uint8))
print('->', f'{W}/vtx/vortex_word_rgba.png')
