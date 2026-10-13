"""Floating shards, shared by the stage loops: rigid pieces of the still drift and tilt in place on a
loop-periodic path (zero offset at frame 0), over a plate where the pieces have been removed.
Motion constants are the approved Vortex ones (vtx_float.py), sized against the cover square side `sq`
as it appears in the output, so every cover floats "1:1, scale considered"."""
import cv2, numpy as np

DRIFT = 0.008            # max drift, fraction of the cover square side
TILT = np.radians(0.9)


def labels_from_plate(master, plate_raw, keep_fn=None, thr=38, min_area=1500):
    """Align+grade the plate to the master, then label the pieces = where master and plate differ."""
    MH, MW = master.shape[:2]
    raw = cv2.resize(plate_raw, (MW, MH), interpolation=cv2.INTER_AREA)
    g = lambda im: cv2.cvtColor(cv2.resize(im, (MW//4, MH//4)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g(raw), g(master))
    raw = cv2.warpAffine(raw, np.float32([[1, 0, dx*4], [0, 1, dy*4]]), (MW, MH), borderMode=cv2.BORDER_REFLECT)
    gain = (cv2.GaussianBlur(master.astype(np.float32), (0, 0), 40)+2)/(cv2.GaussianBlur(raw.astype(np.float32), (0, 0), 40)+2)
    plate = np.clip(raw*np.clip(gain, 0.7, 1.4), 0, 255).astype(np.uint8)
    d = np.abs(cv2.GaussianBlur(master, (0, 0), 3).astype(np.float32)-cv2.GaussianBlur(plate, (0, 0), 3).astype(np.float32)).max(2)
    m = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ff = m.copy(); h, w = ff.shape; fm = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(ff, fm, (0, 0), 2); m = (ff != 2).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, np.int32); k = 0
    for i in range(1, n):
        if st[i, 4] >= min_area and (keep_fn is None or keep_fn(st[i], cen[i])): k += 1; keep[i] = k
    return plate, keep[lab], (dx*4, dy*4)


class Floater:
    def __init__(self, src, plate, lab, sq, seed=7):
        """src/plate/lab at output size; sq = cover square side in output px."""
        H, Wd = src.shape[:2]
        srcf = src.astype(np.float32)
        self.alpha = cv2.GaussianBlur((lab > 0).astype(np.float32), (0, 0), 1.5)[..., None]
        self.plate = plate.astype(np.float32)
        rng = np.random.default_rng(seed); self.pieces = []
        for i in [i for i in np.unique(lab) if i > 0]:
            ys, xs = np.where(lab == i); area = len(ys)
            mg = int(DRIFT*sq*2)+8
            y0, y1, x0, x1 = max(0, ys.min()-mg), min(H, ys.max()+mg+1), max(0, xs.min()-mg), min(Wd, xs.max()+mg+1)
            a = cv2.GaussianBlur((lab[y0:y1, x0:x1] == i).astype(np.float32), (0, 0), 1.2)
            size = np.sqrt(area)/sq*0.5625
            amp = DRIFT*sq*np.clip(0.6+0.08/max(size, 1e-3)*0.05, 0.6, 1.0)
            th = rng.uniform(0, 2*np.pi)
            self.pieces.append(dict(box=(y0, y1, x0, x1), rgba=np.dstack([srcf[y0:y1, x0:x1], a]), area=area,
                                    c=((x1-x0)/2, (y1-y0)/2), d1=amp*np.array([np.cos(th), np.sin(th)]),
                                    d2=0.5*amp*np.array([-np.sin(th), np.cos(th)]),
                                    rot=rng.uniform(-1, 1)*TILT*np.clip(0.02/max(size, 1e-3), 0.3, 1.0)))
        self.pieces.sort(key=lambda p: -p['area'])

    def apply(self, frame, i, length):
        """frame: float32 output frame; pieces are removed (plate) and redrawn at their loop position."""
        ph = 2*np.pi*i/length
        col = frame*(1-self.alpha) + self.plate*self.alpha
        for pc in self.pieces:
            y0, y1, x0, x1 = pc['box']
            off = pc['d1']*np.sin(ph) + pc['d2']*(1-np.cos(ph))
            Mx = cv2.getRotationMatrix2D(pc['c'], np.degrees(pc['rot']*np.sin(ph)), 1.0); Mx[:, 2] += off
            w = cv2.warpAffine(pc['rgba'], Mx, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            a = w[..., 3:4]
            col[y0:y1, x0:x1] = w[..., :3]*a + col[y0:y1, x0:x1]*(1-a)
        return col
