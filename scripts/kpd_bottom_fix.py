"""kpd_bottom_fix.py <W>

"Коли прийде день" only. After build_master.py --seamfix the Gemini extension below the square is
still lighter than the square's last rows. This applies a LUMINANCE-only per-column gain (colour
stays the extension's own, so the green/brown split survives), decaying away from each join, then
darkens the bottom zone (where Spotify/Apple draw player UI) and softens both joins.
The square spans rows 871..3966 of the 5504-high master.
"""
import cv2, numpy as np, sys

W = sys.argv[1]
m = cv2.imread(f'{W}/kpd/master_916.png').astype(np.float32); H, S = m.shape[:2]; y0, y1 = 871, 3966
L = m.mean(2)

def colmean(img):
    return cv2.GaussianBlur(img.mean(0, keepdims=True), (0, 0), sigmaX=80, sigmaY=0.1)[0]

r = np.clip((colmean(L[y0:y0+60])+1)/(colmean(L[y0-80:y0-20])+1), 0.4, 1.8); d = (y0-1-np.arange(y0))[:, None]
m[:y0] = np.clip(m[:y0]*(1+(r[None]-1)*np.exp(-d/500))[..., None], 0, 255)
r = np.clip((colmean(L[y1-60:y1])+1)/(colmean(L[y1+20:y1+80])+1), 0.2, 1.5); d = np.arange(H-y1)[:, None]
m[y1:] = np.clip(m[y1:]*(1+(r[None]-1)*np.exp(-d/900))[..., None], 0, 255)
yy = np.arange(H); z = np.clip((yy-(y1-250))/(H-(y1-250)), 0, 1); m = m*(1-0.55*z**0.9)[:, None, None]
soft = cv2.GaussianBlur(m, (0, 0), sigmaX=0.1, sigmaY=14)
wb = np.maximum(np.exp(-((yy-y1)/22.0)**2), np.exp(-((yy-y0)/16.0)**2))[:, None, None]
cv2.imwrite(f'{W}/kpd/master_916.png', np.clip(soft*wb+m*(1-wb), 0, 255).astype(np.uint8))
print('kpd bottom fixed')
