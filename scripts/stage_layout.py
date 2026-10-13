"""Shared layout for the 16:9 club-stage videos: where the band logo sits and how it is laid over the art."""
import cv2, numpy as np

SIZE = (1920, 1080)
FPS = 30
LOOP_S = 20                       # loop length in seconds
LOGO_W = {'kpd': 0.56, 'bts': 0.56, 'vtx': 0.56, 'slv': 0.56}   # fraction of frame width
# Logo centre height: upper-middle, clear of the band and the stage gear below. BTS sits higher, in the
# archway, so it does not cover the skull crown of the bell.
LOGO_CY = {'kpd': 0.38, 'bts': 0.27, 'slv': 0.38, 'vtx': 0.38}
SHADOW = 0.45                     # soft local darkening behind the logo, for legibility on busy art


def logo_overlay(W, k, size=SIZE, shadow=True):
    """(colour, alpha) at frame size. Alpha includes a soft shadow, so `colour*alpha + frame*(1-alpha)` is final."""
    w, h = size
    lg = cv2.imread(f'{W}/{k}/logo_rgba.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    lw = int(round(LOGO_W[k]*w)); lh = int(round(lg.shape[0]*lw/lg.shape[1]))
    lg = cv2.resize(lg, (lw, lh), interpolation=cv2.INTER_AREA)
    col = np.zeros((h, w, 3), np.float32); a = np.zeros((h, w), np.float32)
    x0 = (w-lw)//2; y0 = int(round(LOGO_CY[k]*h - lh/2))
    col[y0:y0+lh, x0:x0+lw] = lg[..., :3]; a[y0:y0+lh, x0:x0+lw] = lg[..., 3]/255
    sh = SHADOW*cv2.GaussianBlur(cv2.dilate(a, np.ones((9, 9), np.uint8)), (0, 0), w/60) if shadow else np.zeros_like(a)
    at = a + sh*(1-a)
    return col*(a/np.maximum(at, 1e-3))[..., None], at[..., None]
