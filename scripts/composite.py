"""composite.py <W> <key> <motion.mp4> <outdir> <which: spotify|a34|a11>

One deliverable per call (keeps memory low). Inputs per cover in <W>/<key>/:
  master_916.png  tall textless master (3072 wide)    reg_master.json  clean_4k square -> master similarity
  sq_text_4k.png  the cover with lettering            text_layer.png   lettering RGBA on the 4096 square
Motion replaces the still only where the render actually moves (temporal-activity mask), so every
static pixel and all lettering stay the crisp still. Loops cross-fade the tail into the head.
Frames are streamed: only the cross-fade tail is held in memory.
"""
import cv2, numpy as np, json, subprocess, sys, os

W, K, VID, OUT, WHICH = sys.argv[1:6]
CFG = {
    'kpd': dict(top=450, blocks=[((640, 300, 2814, 674), 700, None), ((767, 3409, 2564, 433), 'after', 'unblend')]),
    'bts': dict(top=400, blocks=[((710, 230, 2695, 920), 700, 'outline')]),
    'slv': dict(top=389, blocks=[], cover_patch=True),   # logo taken from the cover itself, at its own place
}[K]
FPS = 30
PLAN = {'spotify': (8*FPS, 1*FPS, 2*FPS), 'a34': (15*FPS, int(1.5*FPS), 0), 'a11': (15*FPS, int(1.5*FPS), 0)}
LEN, XF, OFF = PLAN[WHICH]
# Loop window start inside the Wan take, chosen where the take itself steps smoothly across the seam
# (SLV 1x1: frame 449->450 of the take is a larger jump than usual, so OFF=0 gave a visible seam step).
OFF = {('slv', 'a11'): 11}.get((K, WHICH), OFF)

master = cv2.imread(f'{W}/{K}/master_916.png'); MH, MW = master.shape[:2]
reg = json.load(open(f'{W}/{K}/reg_master.json')); M = np.array(reg['M'], np.float32); sc = reg['scale']
vp = f'{W}/{K}/{VID}'

# ---- pass 1: activity mask (quarter res) + frame 0 + the cross-fade tail frames
cap = cv2.VideoCapture(vp); VW, VH = int(cap.get(3)), int(cap.get(4))
acc = None; acc2 = None; n = 0; f0 = None; tail = {}
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i == 0: f0 = f.copy()
    if OFF+LEN <= i < OFF+LEN+XF: tail[i-OFF-LEN] = f.copy()
    if i % 2 == 0:
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (VW//2, VH//2)), cv2.COLOR_BGR2GRAY), (0, 0), 1.2).astype(np.float64)
        acc = g if acc is None else acc+g; acc2 = g*g if acc2 is None else acc2+g*g; n += 1
    i += 1
cap.release()
assert len(tail) == XF, f'clip too short for {WHICH}: {i} frames'
act = np.sqrt(np.maximum(acc2/n-(acc/n)**2, 0)).astype(np.float32)
m = np.clip((cv2.GaussianBlur(act, (0, 0), 3)-2.5)/(6.0-2.5), 0, 1)   # Wan's texture boil sits below ~2.5
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
m = cv2.GaussianBlur(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))), (0, 0), 5)
m = cv2.resize(m, (VW, VH), interpolation=cv2.INTER_LINEAR)
mv = cv2.resize(master, (VW, VH), interpolation=cv2.INTER_AREA).astype(np.float32)
G = np.clip((cv2.GaussianBlur(mv, (0, 0), 20)+2)/(cv2.GaussianBlur(f0.astype(np.float32), (0, 0), 20)+2), 0.75, 1.35)
kx, ky = VW/MW, VH/MH

DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

def flow_blend(A, B, t):
    """Motion-compensated cross-fade between A (t=0) and B (t=1): both frames are warped to the in-between
    position along the optical flow before mixing, so a swaying flower does not show up twice."""
    h, w = A.shape[:2]
    ga = cv2.cvtColor(cv2.resize(A, (w//2, h//2)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(B, (w//2, h//2)), cv2.COLOR_BGR2GRAY)
    fab = cv2.resize(DIS.calc(ga, gb, None), (w, h))*2      # A(x) ~ B(x + fab(x))
    fba = cv2.resize(DIS.calc(gb, ga, None), (w, h))*2      # B(x) ~ A(x + fba(x))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ia = cv2.remap(A, xx - t*fab[..., 0], yy - t*fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ib = cv2.remap(B, xx - (1-t)*fba[..., 0], yy - (1-t)*fba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    morph = (1-t)*ia.astype(np.float32) + t*ib.astype(np.float32)
    plain = (1-t)*A.astype(np.float32) + t*B.astype(np.float32)
    return morph*FLOWM + plain*(1-FLOWM)

# The flow morph is only needed where a solid object sways (the SLV flower). On smoke and light it warps the
# haze back and forth and reads as a 1-2 s "beating" at the start of the file (Viktor, 2026-10-12), so
# everywhere else the seam is a plain dissolve.
FLOWM = np.zeros((VH, VW, 1), np.float32)
if K == 'slv':
    hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 6) | (hsv[..., 0] > 174)) & (hsv[..., 1] > 170) & (hsv[..., 2] > 140)).astype(np.uint8)
    red[:int(0.3*VH)] = 0; red[int(0.62*VH):] = 0
    n_, lab_, st_, cen_ = cv2.connectedComponentsWithStats(red)
    fx, fy = cen_[1+np.argmax(st_[1:, 4])]
    blob = np.zeros((VH, VW), np.float32)
    cv2.ellipse(blob, (int(fx), int(fy)+60), (110, 170), 0, 0, 360, 1.0, -1)   # flower head + stem
    FLOWM = cv2.GaussianBlur(blob, (0, 0), 25)[..., None]

def frames():
    cap = cv2.VideoCapture(vp)
    for _ in range(OFF): cap.read()
    for j in range(LEN):
        ok, f = cap.read()
        if j < XF:   # loop seam: the tail (end of the window) morphs into the head
            f = flow_blend(f, tail[j], 1 - j/XF)
        yield f.astype(np.float32)*G
    cap.release()

def enc(path, w, h, bitrate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    args = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'medium', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-threads', '6',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:ipratio=1.0:pbratio=1.0:keyint=600:min-keyint=600:scenecut=0', '-movflags', '+faststart']
    if bitrate is None: args += ['-crf', '4']
    else:
        i = args.index('-x264-params'); args[i+1] += ':nal-hrd=cbr:force-cfr=1'
        args += ['-b:v', bitrate, '-minrate', bitrate, '-maxrate', bitrate, '-bufsize', bitrate]
    return subprocess.Popen(args+[path], stdin=subprocess.PIPE)

if WHICH == 'spotify':
    p = enc(f'{OUT}/spotify/{K}_spotify_9x16.mp4', VW, VH, None); mm = m[..., None]
    for f in frames():
        p.stdin.write(np.clip(mv*(1-mm) + f*mm, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()

elif WHICH == 'a34':
    AW, AH = 2048, 2732; top = CFG['top']
    still = cv2.resize(master[top:top+4096], (AW, AH), interpolation=cv2.INTER_AREA).astype(np.float32)
    vy0, vy1 = int(round(top*ky)), int(round((top+4096)*ky))
    m34 = cv2.resize(m[vy0:vy1], (AW, AH), interpolation=cv2.INTER_CUBIC)[..., None]
    y = np.arange(AH)/AH; grad = (1-0.55*np.clip((y-0.60)/0.40, 0, 1)**1.4)[:, None, None]
    tl = cv2.imread(f'{W}/{K}/text_layer.png', cv2.IMREAD_UNCHANGED)
    lay = np.zeros((4096, MW, 4), np.float32); prev = None
    for (bx, by, bw, bh), y0, matte in CFG['blocks']:
        crop = tl[by:by+bh, bx:bx+bw].astype(np.float32)
        if matte == 'outline':
            # Bright letters only (drops the dark tribal ornament), plus the letters' own black outline:
            # a ~12 px ring around them keeps the cover's alpha there.
            L = cv2.cvtColor(crop[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            bright = np.clip((L-55)/35, 0, 1)*crop[..., 3]/255
            ring = cv2.dilate((bright > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
            ring = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 2)
            crop[..., 3] = np.maximum(bright, np.minimum(crop[..., 3]/255, ring))*255
        elif matte == 'unblend':
            # White lettering solved against the clean art, so the cover's dark background between the
            # letters is not carried over; a faint soft shadow keeps it readable on light sky.
            cov = cv2.imread(f'{W}/{K}/sq_text_4k.png').astype(np.float32)[by:by+bh, bx:bx+bw]
            cln = cv2.GaussianBlur(cv2.imread(f'{W}/{K}/clean_4k.png').astype(np.float32)[by:by+bh, bx:bx+bw], (0, 0), 2)
            a = np.clip((cov-cln)/(255-cln+1e-3), 0, 1).min(2)
            a = cv2.GaussianBlur(np.where(a > 0.08, a, 0), (0, 0), 0.7)
            F = np.clip((cov-(1-a[..., None])*cln)/np.maximum(a[..., None], 0.05), 0, 255)
            sh = 0.35*cv2.GaussianBlur(a, (0, 0), 14)
            at = a + sh*(1-a)
            crop = np.dstack([F*(a/np.maximum(at, 1e-3))[..., None], at*255])
        crop = cv2.resize(crop, (int(bw*sc), int(bh*sc)), interpolation=cv2.INTER_AREA)
        x0 = int(round((bx+bw/2)*sc + M[0, 2] - crop.shape[1]/2)); y0 = prev+10 if y0 == 'after' else y0
        lay[y0:y0+crop.shape[0], x0:x0+crop.shape[1]] = crop; prev = y0+crop.shape[0]
    if CFG.get('cover_patch'):
        cov = cv2.imread(f'{W}/{K}/sq_text_4k.png')
        al = tl[..., 3].astype(np.float32)/255
        al = cv2.GaussianBlur(cv2.dilate((al > 0.1).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(np.float32), (0, 0), 30)
        cw = cv2.warpAffine(cov, M, (MW, MH), flags=cv2.INTER_AREA).astype(np.float32)
        aw = cv2.warpAffine(al, M, (MW, MH), flags=cv2.INTER_LINEAR)
        lay = np.dstack([cw, aw*255])[top:top+4096]
    lay = cv2.resize(lay, (AW, AH), interpolation=cv2.INTER_AREA); la = lay[..., 3:4]/255; lc = lay[..., :3]
    cv2.imwrite(f'{W}/{K}/a34_first.png', np.clip(lc*la + still*grad*(1-la), 0, 255).astype(np.uint8))
    p = enc(f'{OUT}/apple/{K}_apple_3x4.mp4', AW, AH, '90M')
    for f in frames():
        mo = cv2.resize(f[vy0:vy1], (AW, AH), interpolation=cv2.INTER_CUBIC)
        o = (still*(1-m34) + mo*m34)*grad
        p.stdin.write(np.clip(lc*la + o*(1-la), 0, 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()

else:  # a11
    S1 = 3840
    base = cv2.resize(cv2.imread(f'{W}/{K}/sq_text_4k.png'), (S1, S1), interpolation=cv2.INTER_AREA).astype(np.float32)
    talpha = cv2.resize(cv2.imread(f'{W}/{K}/text_layer.png', cv2.IMREAD_UNCHANGED)[..., 3], (S1, S1), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    if K == 'bts':  # the ornamental border is part of the cover: keep it static too
        pass
    Mv = np.diag([kx, ky, 1]).astype(np.float32) @ np.vstack([M, [0, 0, 1]]).astype(np.float32) @ np.diag([4096/S1, 4096/S1, 1]).astype(np.float32)
    Mvs = Mv[:2]                                       # output(3840) px -> video px
    m11 = cv2.warpAffine(m, Mvs, (S1, S1), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    m11 = (m11*(1-cv2.dilate(talpha, np.ones((9, 9), np.uint8))))[..., None]
    # the render comes from the clean master; match it to the cover's own grade over moving pixels
    c11 = cv2.warpAffine(mv, Mvs, (S1, S1), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    G11 = np.clip((cv2.GaussianBlur(base, (0, 0), 40)+2)/(cv2.GaussianBlur(c11, (0, 0), 40)+2), 0.75, 1.35)
    p = enc(f'{OUT}/apple/{K}_apple_1x1.mp4', S1, S1, '90M')
    for f in frames():
        mo = cv2.warpAffine(f, Mvs, (S1, S1), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)*G11
        p.stdin.write(np.clip(base*(1-m11) + mo*m11, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
print(K, WHICH, 'done')
