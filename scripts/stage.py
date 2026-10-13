"""stage.py <W> <key> <motion.mp4> <outdir>

16:9 club-stage loop (1920x1080, 30 fps, LOOP_S seconds, silent) for kpd / bts / slv. Vortex uses
`vtx_float.py ... stage` instead. Same compositing as the cover loops: the crisp still everywhere, the
Wan render only where it really moves (temporal-activity mask), colour matched to the still, the loop
seam cross-faded along optical flow, the band logo (no album title) as a static overlay.
The loop window inside the render is chosen where the seam frames already look most alike.
"""
import cv2, numpy as np, subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_layout import SIZE, FPS, LOOP_S, logo_overlay

W, K, VID, OUT = sys.argv[1:5]
LEN, XF = LOOP_S*FPS, int(1.5*FPS)
VW, VH = SIZE
vp = f'{W}/{K}/{VID}'

# ---- pass 1: activity mask, frame 0, and small grey frames for choosing the loop window
cap = cv2.VideoCapture(vp); acc = acc2 = None; n = 0; f0 = None; small = []
while True:
    ok, f = cap.read()
    if not ok: break
    if f.shape[1] != VW: f = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
    if f0 is None: f0 = f.copy()
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (VW//2, VH//2)), cv2.COLOR_BGR2GRAY), (0, 0), 1.2).astype(np.float64)
    acc = g if acc is None else acc+g; acc2 = g*g if acc2 is None else acc2+g*g; n += 1
    small.append(cv2.resize(f, (240, 135), interpolation=cv2.INTER_AREA).astype(np.float32))
cap.release()
N = len(small); assert N >= LEN+XF, f'render too short: {N} frames'
act = np.sqrt(np.maximum(acc2/n-(acc/n)**2, 0)).astype(np.float32)
m = np.clip((cv2.GaussianBlur(act, (0, 0), 3)-2.5)/(6.0-2.5), 0, 1)   # Wan's texture boil sits below ~2.5
# Regions that must move as a whole even where the motion is slow (fractions of the frame, x0, y0, x1, y1):
# KPD's big smoke cloud drifts too slowly to pass the threshold, and a half-frozen cloud reads as broken.
ALWAYS = {'kpd': [(0.0, 0.0, 0.47, 0.50)]}.get(K, [])
for fx0, fy0, fx1, fy1 in ALWAYS:
    h_, w_ = m.shape; r = np.zeros_like(m)
    r[int(fy0*h_):int(fy1*h_), int(fx0*w_):int(fx1*w_)] = 1
    m = np.maximum(m, cv2.GaussianBlur(r, (0, 0), w_/40))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
m = cv2.GaussianBlur(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))), (0, 0), 5)
m = cv2.resize(m, SIZE, interpolation=cv2.INTER_LINEAR)
# Regions that stay the still (Andrew's review): KPD trees (Wan morphs the foliage), BTS sky inside the
# archway (Wan's cloud motion looked odd). BTS lightning is put back on top as pure added light (LIGHT).
STATIC = np.zeros(m.shape, np.float32)
if K == 'kpd':
    STATIC = np.load(f'{W}/kpd/trees_mask_1080.npy').astype(np.float32)
    STATIC[0:600, 1000:] = 1          # right sky: Wan's clouds morphed (~0:10), so it stays the still
# BTS: the whole archway behind the bell, down to the floor, is ONE static backdrop with the lightning added on
# top (Viktor, stage v6: a box ending at y=600 left the lower sky to the render and cut the bolts in half).
LIGHT = {'bts': (670, 0, 1260, 840)}.get(K)          # x0, y0, x1, y1 at 1920x1080
if LIGHT:
    lx0, ly0, lx1, ly1 = LIGHT; STATIC[ly0:ly1, lx0:lx1] = 1
STATIC = cv2.GaussianBlur(STATIC, (0, 0), 6)
m = m*(1-STATIC)
# BTS torches: only actual fire comes from the render (bright warm pixels of the still or of any render frame,
# dilated); the stone around it stays the still, so the torch body does not wobble.
FLAME = None
if False:   # flame-only torch mask: tried 2026-10-13, Andrew preferred Wan's full torch motion (v4)
    hot = np.zeros((VH, VW), np.uint8)
    cap = cv2.VideoCapture(vp); k_ = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if k_ % 5 == 0:
            if f.shape[1] != VW: f = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
            h_ = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            hot |= ((h_[..., 0] < 35) & (h_[..., 1] > 90) & (h_[..., 2] > 170)).astype(np.uint8)
        k_ += 1
    cap.release()
    torches = np.zeros((VH, VW), np.uint8); torches[300:760, 380:700] = 1; torches[300:760, 1220:1520] = 1
    FLAME = cv2.GaussianBlur(cv2.dilate(hot*torches, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(np.float32), (0, 0), 6)
    tb = cv2.GaussianBlur(torches.astype(np.float32), (0, 0), 10)
    m = m*(1-tb)      # Wan's flames morph into blobs: the torches are animated procedurally below instead
# Procedural fire (BTS torches): the cover's own flame, warped by an upward-travelling displacement field and
# flickered in brightness. Every term has an integer number of cycles per loop, so it is seamless; no morphing.
FIRE = {}.get(K, [])   # procedural torches (bts) tried 2026-10-13 and rejected: read as constant morphing
fire_setup = []
for fx0, fy0, fx1, fy1 in FIRE:
    sb = cv2.resize(cv2.imread(f'{W}/{K}/wide_master.png'), SIZE, interpolation=cv2.INTER_AREA)[fy0:fy1, fx0:fx1]
    h_ = cv2.cvtColor(sb, cv2.COLOR_BGR2HSV)
    hot = ((h_[..., 0] < 35) & (h_[..., 1] > 80) & (h_[..., 2] > 150)).astype(np.uint8)
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats(hot)
    hot = np.isin(lab_, [k for k in range(1, n_) if st_[k, 4] > 300]).astype(np.uint8)
    ys_, xs_ = np.where(hot > 0); base_y = ys_.max(); top_y = ys_.min()
    fm = cv2.GaussianBlur(cv2.dilate(hot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32), (0, 0), 8)
    gy, gx = np.mgrid[0:fy1-fy0, 0:fx1-fx0].astype(np.float32)
    rise = np.clip((base_y-gy)/max(1, base_y-top_y), 0, 1.4)                      # 0 at the flame base, 1 at the tip
    fire_setup.append(dict(box=(fx0, fy0, fx1, fy1), src=sb.astype(np.float32), fm=fm[..., None], gx=gx, gy=gy, rise=rise))
def fire_frame(fs, j, L):
    t = 2*np.pi*j/L
    gx, gy, r = fs['gx'], fs['gy'], fs['rise']
    dx = r*(5.0*np.sin(gy/23.0 + 31*t) + 2.5*np.sin(gy/9.0 + 53*t + 1.3) + 1.5*np.sin(gx/15.0 + 47*t))
    dy = r*(3.0*np.sin(gy/17.0 + 41*t + 0.7) + 1.5*np.sin(gx/11.0 + 59*t)) + 2.0*r*(1+np.sin(37*t))
    w = cv2.remap(fs['src'], (gx+dx).astype(np.float32), (gy+dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    flick = 1 + 0.10*np.sin(29*t) + 0.06*np.sin(67*t + 0.4) + 0.04*np.sin(97*t + 2.1)
    return np.clip(w*flick, 0, 255)
# KPD birds: Wan flies them around, merges them and paints some pale, whatever the prompt says. In the bird
# area only the CLOUDS come from the render (per frame: pale low-saturation pixels of the render or the still,
# dilated); the open blue sky, birds included, is the still, so the cover's birds stay put and nothing else flies.
BIRDS = {'kpd': (0.50, 0.0, 1.0, 0.45)}.get(K)
if BIRDS:
    bx0, by0, bx1, by1 = int(BIRDS[0]*VW), int(BIRDS[1]*VH), int(BIRDS[2]*VW), int(BIRDS[3]*VH)
    BOX = np.zeros(m.shape, np.float32); BOX[by0:by1, bx0:bx1] = 1
    BOX = cv2.GaussianBlur(BOX, (0, 0), 25)
    _still_box = cv2.resize(cv2.imread(f'{W}/{K}/wide_master.png'), SIZE, interpolation=cv2.INTER_AREA)
def cloudness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    return (np.clip((hsv[..., 2]-140)/40, 0, 1)*np.clip((80-hsv[..., 1])/40, 0, 1))
CLOUD_STILL = cv2.dilate(cloudness(_still_box), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) if BIRDS else None
SPECK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
def clean_cloud_birds(f):
    """Remove the render's birds that cross clouds or their edges (where bird_gate lets the render through):
    small, compact, strongly darker blobs, filled with their surroundings (normalised blur)."""
    if not BIRDS: return f
    f = f.copy(); reg = f[by0:by1, bx0:bx1]
    u8 = np.clip(reg, 0, 255).astype(np.uint8)
    dk = (cv2.morphologyEx(u8, cv2.MORPH_CLOSE, SPECK).astype(np.int16)-u8.astype(np.int16)).max(2) > 35
    n, lab, st, _ = cv2.connectedComponentsWithStats(dk.astype(np.uint8))
    # bird-sized and compact only: cloud edges give long thin components and are left alone
    mk = np.isin(lab, [k for k in range(1, n) if st[k, 4] <= 700 and st[k, 2] <= 70 and st[k, 3] <= 35]).astype(np.uint8)
    if mk.any():
        mk = cv2.dilate(mk, np.ones((5, 5), np.uint8)); keep = (1-mk).astype(np.float32)
        fill = cv2.GaussianBlur(reg.astype(np.float32)*keep[..., None], (0, 0), 6)/np.maximum(cv2.GaussianBlur(keep, (0, 0), 6), 1e-3)[..., None]
        reg[mk > 0] = fill[mk > 0]
    return f
def bird_gate(f):
    """Per-frame multiplier for the motion mask inside the bird box: 1 on clouds, 0 on open sky."""
    c = np.maximum(CLOUD_STILL, cv2.dilate(cloudness(np.clip(f, 0, 255).astype(np.uint8)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))))
    c = cv2.GaussianBlur(c, (0, 0), 6)
    return (1-BOX) + BOX*c
ms = cv2.resize(m, (240, 135)) > 0.3
# Discrete events (BTS lightning) must not fall into either seam window: a flash cut by the seam shows up as
# a slowly dissolving ghost bolt. If the nominal length cannot avoid them, the loop is shortened (down to 70%).
# Storm glow lingers for most of the clip, so: a short seam (0.5 s, like a natural lightning decay) and both
# seam windows in quiet frames (sky luminance close to its 10th-percentile baseline). Only for FLASHY covers.
FLASHY = {'bts'}
EVENT = np.zeros(N, bool)
if K in FLASHY:
    XF = int(0.5*FPS)
    Ls = np.stack([x.max(2) for x in small]); base = np.percentile(Ls, 10, axis=0)
    sky = np.zeros(Ls.shape[1:], bool); sky[0:int(0.67*Ls.shape[1]), int(0.25*Ls.shape[2]):int(0.75*Ls.shape[2])] = True
    score = np.array([np.clip(l-base-6, 0, None)[sky].mean() for l in Ls])
    EVENT = score > np.percentile(score, 10) + 1.0
def clean(o, L):
    return not EVENT[o:o+XF].any() and not EVENT[o+L:o+L+XF].any()
# Wan spends the first ~1 s of a take turning the still into motion (BTS: the torch flames visibly morph out of
# the painted flame), so the loop never starts there.
MIN_OFF = {'bts': 36}.get(K, 0)
cands = [(L, o) for L in range(LEN, int(LEN*0.7)-1, -1) for o in range(MIN_OFF, N-L-XF+1) if clean(o, L)] if EVENT.any() else []
if cands:
    LEN = cands[0][0]; pool = [o for L, o in cands if L == LEN]
else:
    pool = list(range(MIN_OFF, N-LEN-XF+1))
cost = {o: np.mean([np.abs(small[o+j]-small[o+LEN+j])[ms].mean() for j in range(0, XF, 5)]) for o in pool}
OFF = min(cost, key=cost.get)
print(K, 'events', int(EVENT.sum()), 'frames | loop', round(LEN/FPS, 2), 's from frame', OFF, 'seam cost', round(cost[OFF], 2))

# Slow global light drift inside the window (Wan tends to brighten a scene over 20 s) would snap back at
# the seam. Fit a per-channel linear trend of the moving region's mean and divide it out, so the loop's
# end matches its start; local flicker (fire, sparks) is untouched.
means = np.array([[small[OFF+j][..., c][ms].mean() for c in range(3)] for j in range(LEN)])
tj = np.arange(LEN); fit = [np.polyfit(tj, means[:, c], 1) for c in range(3)]
trend = np.stack([np.polyval(fit[c], tj) for c in range(3)], 1)
DETREND = (trend[0]/trend).astype(np.float32)               # per frame, per channel
# The tail frames sit at times LEN..LEN+XF of the render: extrapolate the trend to them.
tt = np.arange(LEN, LEN+XF)
TAIL_DETREND = (trend[0]/np.stack([np.polyval(fit[c], tt) for c in range(3)], 1)).astype(np.float32)
print(K, 'light drift over loop (%):', np.round((trend[-1]/trend[0]-1)*100, 1))

master = cv2.resize(cv2.imread(f'{W}/{K}/wide_master.png'), SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
G = np.clip((cv2.GaussianBlur(master, (0, 0), 20)+2)/(cv2.GaussianBlur(f0.astype(np.float32), (0, 0), 20)+2), 0.75, 1.35)
lc, la = logo_overlay(W, K)

# ---- procedural layers drawn from the still, all periodic in LEN (seamless) and still at frame 0
PROC = []
import importlib.util
from float_shards import Floater, labels_from_plate
if K == 'bts':
    # Fragments around the bell float exactly like the Vortex glass (same constants, scaled to the cover).
    import json as _json
    _sq = 4096*_json.load(open(f'{W}/bts/wide_reg.json'))['scale']*VW/cv2.imread(f'{W}/bts/wide_master.png').shape[1]
    _plate, _lab, _ = labels_from_plate(master.astype(np.uint8), cv2.imread(f'{W}/bts/wide_plate_raw.png'), thr=30, min_area=40)
    _fl = Floater(master.astype(np.uint8), _plate, _lab, _sq)
    print(K, 'floating fragments', int(_lab.max()))
    PROC.append(lambda o, j: _fl.apply(o, j, LEN))
if K == 'kpd':
    yy_, xx_ = np.mgrid[0:VH, 0:VW].astype(np.float32)
    # The rift "breathes" as its own layer over the seam (Andrew, v8/v9: the sides must not deform). Rift alpha =
    # work/kpd/rift_alpha_1080.npy (see below). The layer only
    # EXPANDS outward from the centre line (0..9%) and is laid over the untouched master, so the sides are
    # covered a little more and then uncovered again, never warped. The red glow pulses down the crack.
    # Rift outline from a Gemini segmentation (prompts/p_mask_kpd_rift.txt -> work/kpd/rift_magenta.png), eroded 2 px
    # so no ground is carried along; the centre line is the per-row middle of that outline.
    cx = np.load(f'{W}/kpd/rift_center_1080.npy').astype(np.float32)[:, None]
    dist = xx_ - cx
    RA = np.load(f'{W}/kpd/rift_alpha_1080.npy').astype(np.float32)
    hsv_ = cv2.cvtColor(master.astype(np.uint8), cv2.COLOR_BGR2HSV)
    glow = (((hsv_[..., 0] < 12) | (hsv_[..., 0] > 168)) & (hsv_[..., 1] > 110) & (hsv_[..., 2] > 60)).astype(np.float32)
    glow = cv2.GaussianBlur(glow, (0, 0), 3)*RA
    def rift(o, j):
        t = j/LEN
        s_ = 0.09*(0.5 - 0.5*np.cos(2*np.pi*5*t))
        xs = (cx + dist/(1 + s_)).astype(np.float32)
        lay = cv2.remap(master, xs, yy_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        a_ = cv2.remap(RA, xs, yy_, cv2.INTER_LINEAR)[..., None]
        gw = cv2.remap(glow, xs, yy_, cv2.INTER_LINEAR)[..., None]
        pulse = 0.45*(0.5 + 0.5*np.sin(2*np.pi*7*t - yy_[..., None]/140))
        lay = lay*(1 + gw*pulse)
        return lay*a_ + o*(1-a_)
    PROC.append(rift)
    # Birds: the cover's flock, redrawn as flapping silhouettes at the same places, sizes and colour (Andrew, v8:
    # squashing the painted sprite read as morphing). Each bird = body + two wings hinged at the body; the wings
    # beat up and down (elbow + tip), ~1.2 beats/s, each bird in its own phase. Flock drift as in v8.
    bx0_, by0_, bx1_, by1_ = 1180, 40, 1370, 190
    reg = master[by0_:by1_, bx0_:bx1_].astype(np.uint8)
    cl_ = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bm = ((cl_.astype(np.int16)-reg.astype(np.int16)).max(2) > 25).astype(np.uint8)
    nb, lb, sb, cb = cv2.connectedComponentsWithStats(bm)
    sky_plate = master.copy()
    sky_plate[by0_:by1_, bx0_:bx1_] = cv2.inpaint(reg, cv2.dilate(bm, np.ones((7, 7), np.uint8)), 6, cv2.INPAINT_TELEA).astype(np.float32)
    rng_ = np.random.default_rng(3); birds = []
    for k_ in range(1, nb):
        if sb[k_, 4] < 8: continue
        x, y, w, h = sb[k_, :4]
        col = np.median(reg[lb == k_].reshape(-1, 3), axis=0).astype(np.float32)
        ys_, xs_ = np.where(lb == k_)
        tilt = float(np.polyfit(xs_, ys_, 1)[0]) if np.ptp(xs_) > 3 else 0.0
        birds.append(dict(cx=bx0_+cb[k_][0], cy=by0_+cb[k_][1], span=max(w, 10)*1.05, col=col,
                          tilt=np.clip(tilt, -0.5, 0.5), ph=rng_.uniform(0, 2*np.pi, 4), th=max(1.4, h*0.22)))
    print(K, 'birds', len(birds))
    BOXB = (by0_-25, by1_+25, bx0_-40, bx1_+40)
    SS = 4                                            # supersampling for antialiased silhouettes
    def draw_bird(canvas_a, b_, ox, oy, beat):
        X0b, Y0b = BOXB[2], BOXB[0]
        cx_, cy_ = (b_['cx']+ox-X0b)*SS, (b_['cy']+oy-Y0b)*SS
        half = b_['span']/2*SS; up = np.sin(beat)              # +1 wings up, -1 wings down
        elbow_y, tip_y = -0.30*half*up, -0.62*half*up + 0.10*half
        pts_ = []
        for sgn in (-1, 1):
            pts_.append(np.array([[cx_, cy_], [cx_+sgn*0.45*half, cy_+elbow_y], [cx_+sgn*half, cy_+tip_y]]))
        c_, s_ = np.cos(np.arctan(b_['tilt'])), np.sin(np.arctan(b_['tilt']))
        for wing in pts_:
            d = wing - [cx_, cy_]; wing[:] = np.stack([cx_ + d[:, 0]*c_ - d[:, 1]*s_, cy_ + d[:, 0]*s_ + d[:, 1]*c_], 1)
            th = int(round(b_['th']*SS))
            cv2.line(canvas_a, tuple(np.int32(wing[0])), tuple(np.int32(wing[1])), 1.0, th, cv2.LINE_AA)
            cv2.line(canvas_a, tuple(np.int32(wing[1])), tuple(np.int32(wing[2])), 1.0, max(1, th*2//3), cv2.LINE_AA)
        cv2.ellipse(canvas_a, (int(cx_), int(cy_)), (int(b_['th']*SS*1.4), int(b_['th']*SS*0.9)), np.degrees(np.arctan(b_['tilt'])), 0, 360, 1.0, -1, cv2.LINE_AA)
    def fly(o, j):
        t = j/LEN; Y0b, Y1b, X0b, X1b = BOXB
        o[Y0b:Y1b, X0b:X1b] = sky_plate[Y0b:Y1b, X0b:X1b]
        gx, gy = 14*np.sin(2*np.pi*t), 6*np.sin(4*np.pi*t)
        for b_ in birds:
            p0, p1, p2, p3 = b_['ph']
            ox = gx + 4*np.sin(2*np.pi*2*t + p0); oy = gy + 3*np.sin(2*np.pi*3*t + p1)
            beat = 2*np.pi*12*t + p2                     # 12 beats per 20 s loop (~0.6/s), slower per Andrew
            oy += 1.2*np.sin(beat + np.pi)                  # the body bobs against the wing beat
            A_ = np.zeros(((Y1b-Y0b)*SS, (X1b-X0b)*SS), np.float32)
            draw_bird(A_, b_, ox, oy, beat)
            a_ = cv2.resize(A_, (X1b-X0b, Y1b-Y0b), interpolation=cv2.INTER_AREA)[..., None]
            o[Y0b:Y1b, X0b:X1b] = b_['col']*a_ + o[Y0b:Y1b, X0b:X1b]*(1-a_)
        return o
    PROC.append(fly)
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

PLAIN = np.zeros(m.shape, np.float32)          # where the seam is a plain dissolve instead of a flow morph
for fx0, fy0, fx1, fy1 in ALWAYS:
    h_, w_ = PLAIN.shape; PLAIN[int(fy0*h_):int(fy1*h_), int(fx0*w_):int(fx1*w_)] = 1
PLAIN = cv2.GaussianBlur(PLAIN, (0, 0), PLAIN.shape[1]/40)[..., None] if ALWAYS else None
if K == 'bts':
    # BTS has no swaying solid object, and the flow morph turned the torch flames into a 0.5 s morph at the
    # loop point (Andrew, v4): the whole seam is a plain dissolve.
    PLAIN = np.ones((VH, VW, 1), np.float32)

def flow_blend(A, B, t):
    """Motion-compensated cross-fade between A (t=0) and B (t=1)."""
    h, w = A.shape[:2]
    u8 = lambda im: np.clip(im, 0, 255).astype(np.uint8)
    ga = cv2.cvtColor(cv2.resize(u8(A), (w//2, h//2)), cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(cv2.resize(u8(B), (w//2, h//2)), cv2.COLOR_BGR2GRAY)
    fab = cv2.resize(DIS.calc(ga, gb, None), (w, h))*2; fba = cv2.resize(DIS.calc(gb, ga, None), (w, h))*2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ia = cv2.remap(A, xx-t*fab[..., 0], yy-t*fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ib = cv2.remap(B, xx-(1-t)*fba[..., 0], yy-(1-t)*fba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = (1-t)*ia.astype(np.float32) + t*ib.astype(np.float32)
    if PLAIN is not None:
        # Steady directional motion (KPD smoke) would visibly run BACKWARDS under a flow morph from the
        # tail position to the head position; a dissolve keeps it moving forward (soft smoke hides the overlap).
        out = out*(1-PLAIN) + ((1-t)*A.astype(np.float32) + t*B.astype(np.float32))*PLAIN
    return out

# ---- lightning as added light: (render - its own running minimum over +-0.5 s) keeps only short bright events;
# slow cloud drift barely registers in a 1 s window and is cut by the 10-level floor.
if LIGHT:
    cap = cv2.VideoCapture(vp); crops = []
    while True:
        ok, f = cap.read()
        if not ok: break
        if f.shape[1] != VW: f = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
        crops.append(f[ly0:ly1, lx0:lx1])
    cap.release(); crops = np.stack(crops)
    RW = 15
    runmin = np.stack([crops[max(0, i-RW):i+RW+1].min(0) for i in range(len(crops))])
    LFEATHER = np.zeros((ly1-ly0, lx1-lx0), np.float32); LFEATHER[2:-40, 22:-22] = 1
    LFEATHER = cv2.GaussianBlur(LFEATHER, (0, 0), 15)[..., None]
    def raw_light(i):
        return np.clip(crops[i].astype(np.float32)-runmin[i].astype(np.float32)-10, 0, None)*LFEATHER
    # Wan redraws the bolt every frame, so a played-back flash morphs. Each flash (run of lit frames) is
    # replaced by ONE frozen frame, its brightest, played with a strike envelope: flash, flicker, decay.
    energy = np.array([raw_light(i).mean() for i in range(len(crops))])
    lit = energy > max(1.0, np.percentile(energy, 50)*2)
    ENV = [0.7, 1.0, 0.45, 0.9, 0.6, 0.38, 0.24, 0.15, 0.09, 0.05, 0.02]
    PEAK = {}; GAIN = np.zeros(len(crops), np.float32); SRC = np.zeros(len(crops), np.int32)
    idx = np.where(lit)[0]
    for run in (np.split(idx, np.where(np.diff(idx) > 6)[0]+1) if len(idx) else []):
        pk = int(run[np.argmax(energy[run])]); PEAK[pk] = raw_light(pk)
        for e, g in enumerate(ENV):
            if run[0]+e < len(crops): GAIN[run[0]+e] = g; SRC[run[0]+e] = pk
    print(K, 'lightning strikes', len(PEAK))
    def light(i):
        return PEAK[SRC[i]]*GAIN[i] if GAIN[i] > 0 else 0.0

# ---- pass 2: render
cap = cv2.VideoCapture(vp); frames = []
for i in range(OFF+LEN+XF):
    ok, f = cap.read()
    if f.shape[1] != VW: f = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
    if i >= OFF+LEN: frames.append(f)          # the tail that morphs into the head
cap.release()
os.makedirs(OUT, exist_ok=True)
path = f'{OUT}/{K}_stage_16x9.mp4'
p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{VW}x{VH}', '-r', str(FPS), '-i', '-',
                      '-an', '-c:v', 'libx264', '-preset', 'slow', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-crf', '6',
                      '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                      '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:ipratio=1.0:pbratio=1.0:keyint=900:min-keyint=900:scenecut=0',
                      '-movflags', '+faststart', path], stdin=subprocess.PIPE)
cap = cv2.VideoCapture(vp)
for _ in range(OFF): cap.read()
mm = m[..., None]
for j in range(LEN):
    ok, f = cap.read()
    if f.shape[1] != VW: f = cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA)
    if j < XF:  # the tail frame (time LEN+j) is de-trended with its own, extrapolated gain
        f = flow_blend(f.astype(np.float32)*DETREND[j], frames[j].astype(np.float32)*TAIL_DETREND[j], 1 - j/XF)
    else:
        f = f.astype(np.float32)*DETREND[j]
    if BIRDS: f = clean_cloud_birds(f)
    mj = mm*bird_gate(f*G)[..., None] if BIRDS else mm
    o = master*(1-mj) + f*G*mj
    for layer in PROC:
        o = layer(o, j)
    for fs in fire_setup:
        fx0, fy0, fx1, fy1 = fs['box']
        o[fy0:fy1, fx0:fx1] = fire_frame(fs, j, LEN)*fs['fm'] + o[fy0:fy1, fx0:fx1]*(1-fs['fm'])
    if LIGHT:
        w_ = j/XF if j < XF else 1.0
        add = w_*light(OFF+j) + (1-w_)*light(OFF+LEN+j) if j < XF else light(OFF+j)
        o[ly0:ly1, lx0:lx1] += add
    p.stdin.write(np.clip(lc*la + o*(1-la), 0, 255).astype(np.uint8).tobytes())
cap.release(); p.stdin.close(); p.wait()
print('->', path)
