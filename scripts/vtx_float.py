"""vtx_float.py <W> <outdir> <which: spotify|a34|a11|stage|preview|masks>

Vortex v2: the v1 flashlight sweep PLUS the glass shards floating gently in place.

Layers:
  plate      vtx/plate_916.png, the stairwell with every floating shard removed (Gemini, prompts/p_plate_vtx.txt),
             aligned and graded to the master. Used ONLY under the shards, where a moving shard uncovers it.
  shards     master pixels where master and plate differ (filled, slightly dilated to take the glow halo),
             split into connected components. Each component is a rigid piece that drifts by a few pixels and
             tilts a fraction of a degree. Its photo (faces, eyes) moves with it untouched, so nothing morphs.
Every motion term is periodic over the loop and zero at frame 0, so frame 0 is the cover and the loop is
seamless. Lettering is a static overlay. The bottom-left photo collage stays in the plate, so it does not move.
"""
import cv2, numpy as np, json, subprocess, sys, os

W, OUT, WHICH = sys.argv[1:4]
FPS = 30
SWEEP, ANGLE, WIDTH = 1.1, np.radians(28), 0.085
APPLE_SWEEPS = [1.5, 9.0]
BG_S_DEFAULT = 0.6     # stage background darkening strength: 0.6 of the reference (≈24% darker), the least that keeps the logo clear (see README)
# Shard motion is sized against the cover square as it appears in each output (sq px), so every format
# floats exactly like the approved 16:9 stage loop (there the square is 1080 px and the drift 8.6 px).
DRIFT = 0.008         # max shard drift, fraction of the square side
TILT = np.radians(0.9)
STAGE = WHICH == 'stage'           # 16:9 club-stage version: wide master, its own plate/labels, band logo only
TAG = 'wide' if STAGE else '916'
master = cv2.imread(f'{W}/vtx/{"wide_master" if STAGE else "master_916"}.png'); MH, MW = master.shape[:2]
reg = json.load(open(f'{W}/vtx/{"wide_reg" if STAGE else "reg_master"}.json')); M = np.array(reg['M'], np.float32); sc = reg['scale']
tl = cv2.imread(f'{W}/vtx/text_layer.png', cv2.IMREAD_UNCHANGED)


def prepare_plate_and_labels():
    """Plate aligned/graded to the master, and the shard label map, both in master coordinates (cached)."""
    pp, lp = f'{W}/vtx/plate_{TAG}.png', f'{W}/vtx/shard_labels_{TAG}.png'
    if os.path.exists(pp) and os.path.exists(lp):
        return cv2.imread(pp), cv2.imread(lp, cv2.IMREAD_UNCHANGED).astype(np.int32)
    raw = cv2.imread(f'{W}/vtx/{"wide_plate_raw" if STAGE else "plate_raw"}.png')
    raw = cv2.resize(raw, (MW, MH), interpolation=cv2.INTER_AREA)
    g = lambda im: cv2.cvtColor(cv2.resize(im, (MW//4, MH//4)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g(raw), g(master))
    raw = cv2.warpAffine(raw, np.float32([[1, 0, dx*4], [0, 1, dy*4]]), (MW, MH), borderMode=cv2.BORDER_REFLECT)
    gain = (cv2.GaussianBlur(master.astype(np.float32), (0, 0), 40)+2)/(cv2.GaussianBlur(raw.astype(np.float32), (0, 0), 40)+2)
    plate = np.clip(raw*np.clip(gain, 0.7, 1.4), 0, 255).astype(np.uint8)
    d = np.abs(cv2.GaussianBlur(master, (0, 0), 3).astype(np.float32)-cv2.GaussianBlur(plate, (0, 0), 3).astype(np.float32)).max(2)
    m = cv2.morphologyEx((d > 38).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    ff = m.copy(); h, w = ff.shape; fm = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(ff, fm, (0, 0), 2); m = ((ff != 2)).astype(np.uint8)          # fill holes
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    # Stay static: anything centred in the spiral's funnel (the plate redraws the stairwell there, so a
    # "shard" there would tear the spiral) and the bottom-left photo collage, which is scenery.
    axis = cv2.transform(np.float32([[[0.506*4096, 0.477*4096]]]), M)[0, 0]; side = 4096*sc
    keep = np.zeros(n, np.int32); k = 0
    for i in range(1, n):
        cx, cy = cen[i]
        in_funnel = np.hypot(cx-axis[0], cy-axis[1]) < 0.11*side
        in_collage = (cx < 0.5*MW and cy > 0.62*MH) if STAGE else (cx < 0.48*MW and 0.57*MH < cy < 0.76*MH)
        if st[i, 4] > 1500 and not in_funnel and not in_collage: k += 1; keep[i] = k
    lab = keep[lab]
    cv2.imwrite(pp, plate); cv2.imwrite(lp, lab.astype(np.uint16))
    print('plate shift', round(dx*4, 1), round(dy*4, 1), 'shards', k)
    return plate, lab


def edge_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    return cv2.GaussianBlur(np.clip((hsv[..., 2]-170)/60, 0, 1)*np.clip((100-hsv[..., 1])/70, 0, 1), (0, 0), 1.2)


def enc(path, w, h, bitrate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    args = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'medium', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-threads', '6',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:ipratio=1.0:pbratio=1.0:keyint=600:min-keyint=600:scenecut=0',
            '-movflags', '+faststart']
    if bitrate is None: args += ['-crf', '4']
    else:
        i = args.index('-x264-params'); args[i+1] += ':nal-hrd=cbr:force-cfr=1'
        args += ['-b:v', bitrate, '-minrate', bitrate, '-maxrate', bitrate, '-bufsize', bitrate]
    return subprocess.Popen(args+[path], stdin=subprocess.PIPE)


def render(src, plate, lab, length, starts, post, path, bitrate, edge_block=None, frames_only=None, sq=None):
    """src/plate/lab are already at output size. starts: seconds at which a sweep beam enters."""
    H, Wd = src.shape[:2]
    srcf, platef = src.astype(np.float32), plate.astype(np.float32)
    E = edge_mask(src)
    if edge_block is not None: E *= (1-edge_block)
    glow = cv2.GaussianBlur(E, (0, 0), max(2, Wd/300))
    bloom = cv2.GaussianBlur(E, (0, 0), max(6, Wd/70)); bloom /= max(1e-6, float(np.percentile(bloom, 99.5)))
    yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
    D = np.hypot(H, Wd); proj = ((xx*np.cos(ANGLE) + yy*np.sin(ANGLE))/D).astype(np.float32)
    lo, hi = float(proj.min())-3*WIDTH, float(proj.max())+3*WIDTH
    del xx, yy
    tint = np.array([255, 250, 235], np.float32)
    # Layers that move with the shards: colour, alpha, edge light (E, glow, bloom).
    shard_alpha = cv2.GaussianBlur((lab > 0).astype(np.float32), (0, 0), 1.5)
    base = srcf*(1-shard_alpha[..., None]) + platef*shard_alpha[..., None]
    baseE = np.dstack([E, glow, bloom])*(1-shard_alpha[..., None])
    rng = np.random.default_rng(7)
    ids = [i for i in np.unique(lab) if i > 0]
    pieces = []
    for i in ids:
        ys, xs = np.where(lab == i); area = len(ys)
        mg = int(DRIFT*sq*2)+8
        y0, y1, x0, x1 = max(0, ys.min()-mg), min(H, ys.max()+mg+1), max(0, xs.min()-mg), min(Wd, xs.max()+mg+1)
        a = cv2.GaussianBlur((lab[y0:y1, x0:x1] == i).astype(np.float32), (0, 0), 1.2)
        rgba = np.dstack([srcf[y0:y1, x0:x1], a, E[y0:y1, x0:x1], glow[y0:y1, x0:x1], bloom[y0:y1, x0:x1]])
        size = np.sqrt(area)/sq*0.5625
        amp = DRIFT*sq*np.clip(0.6+0.08/max(size, 1e-3)*0.05, 0.6, 1.0)     # small pieces drift a touch more
        th = rng.uniform(0, 2*np.pi)
        pieces.append(dict(box=(y0, y1, x0, x1), rgba=rgba, area=area, c=((x1-x0)/2, (y1-y0)/2),
                           d1=amp*np.array([np.cos(th), np.sin(th)]), d2=0.5*amp*np.array([-np.sin(th), np.cos(th)]),
                           rot=rng.uniform(-1, 1)*TILT*np.clip(0.02/max(size, 1e-3), 0.3, 1.0)))
    pieces.sort(key=lambda p: -p['area'])                                # big pieces first, small on top
    p = None if frames_only is not None else enc(path, Wd, H, bitrate)
    out_frames = []
    rng_frames = range(length) if frames_only is None else frames_only
    for i in rng_frames:
        ph = 2*np.pi*i/length
        col = base.copy(); el = baseE.copy()
        for pc in pieces:
            y0, y1, x0, x1 = pc['box']
            off = pc['d1']*np.sin(ph) + pc['d2']*(1-np.cos(ph))
            Mx = cv2.getRotationMatrix2D(pc['c'], np.degrees(pc['rot']*np.sin(ph)), 1.0)
            Mx[:, 2] += off
            w = cv2.warpAffine(pc['rgba'], Mx, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            a = w[..., 3:4]
            col[y0:y1, x0:x1] = w[..., :3]*a + col[y0:y1, x0:x1]*(1-a)
            el[y0:y1, x0:x1] = w[..., 4:7] + el[y0:y1, x0:x1]*(1-a)
        f = col
        t = next((i/FPS - st for st in starts if 0 <= i/FPS - st <= SWEEP), None)
        if t is not None:
            s = t/SWEEP; s = s*s*(3-2*s)
            b = np.exp(-((proj-(lo+(hi-lo)*s))/WIDTH)**2)
            f = f*(1 + 1.3*el[..., 0]*b)[..., None] + ((1.6*el[..., 1] + 0.55*el[..., 2])*b)[..., None]*tint + (0.10*b)[..., None]*f
        if post is not None: f = post(f)
        f = np.clip(f, 0, 255).astype(np.uint8)
        if p is None: out_frames.append(f)
        else: p.stdin.write(f.tobytes())
    if p is not None:
        p.stdin.close(); p.wait(); print('->', path)
    return out_frames


plate, lab = prepare_plate_and_labels()
if WHICH == 'masks':
    sys.exit(0)
if STAGE:
    from stage_layout import SIZE, LOOP_S, logo_overlay
    s_, p_, l_ = (cv2.resize(master, SIZE, interpolation=cv2.INTER_AREA), cv2.resize(plate, SIZE, interpolation=cv2.INTER_AREA),
                  cv2.resize(lab, SIZE, interpolation=cv2.INTER_NEAREST))
    # Look matched to Andrew's own Vortex backdrop (Drive: vortex-bg-min-logo-3m.mp4), measured 2026-10-13:
    # background = cover x per-channel gain/offset (BGR 0.604/-3.1, 0.623/-2.7, 0.573/-7.2: ~45% darker, a touch
    # teal); logo drop shadow to the lower right, offset 10 px, blur sigma 8 px, opacity 0.42 at a 1280 px wide
    # logo (scaled to our logo width), almost nothing above-left.
    from stage_layout import LOGO_W
    lc, la = logo_overlay(W, 'vtx', shadow=False)
    k_ = LOGO_W['vtx']*SIZE[0]/1280
    a0 = la[..., 0]
    off = 10*k_
    sh = cv2.GaussianBlur(cv2.warpAffine(a0, np.float32([[1, 0, off], [0, 1, off]]), SIZE), (0, 0), 8*k_)*0.42
    sh = (sh*(1-a0))[..., None]
    # BG_S scales the darkening: 1 = exactly the reference backdrop, 0 = no darkening (VTX_BG_S env overrides).
    BG_S = float(os.environ.get('VTX_BG_S', BG_S_DEFAULT))
    BG_G = 1 - BG_S*(1 - np.array([0.604, 0.623, 0.573], np.float32)); BG_O = BG_S*np.array([-3.1, -2.7, -7.2], np.float32)
    post = lambda f: lc*la + np.clip(f*BG_G+BG_O, 0, 255)*(1-sh)*(1-la)
    eb = cv2.dilate(a0, np.ones((9, 9), np.uint8))
    if os.environ.get('VTX_PROBE'):   # undarkened sample frames for the legibility analysis (incl. sweep peaks)
        fr = render(s_, p_, l_, LOOP_S*FPS, [1.5, 8.0, 14.5], None, None, None, frames_only=list(range(0, LOOP_S*FPS, 15)) + [55, 62, 250, 257, 445, 452],
                    sq=4096*sc*SIZE[0]/MW, edge_block=eb)
        np.save(f'{W}/vtx/probe_frames.npy', np.stack(fr)); np.save(f'{W}/vtx/probe_logo.npy', np.dstack([lc, la, sh]))
        sys.exit(0)
    render(s_, p_, l_, LOOP_S*FPS, [1.5, 8.0, 14.5], post, f'{OUT}/vtx_stage_16x9.mp4', None, sq=4096*sc*SIZE[0]/MW, edge_block=eb)
elif WHICH in ('spotify', 'preview'):
    sz = (1080, 1920)
    s_, p_, l_ = cv2.resize(master, sz, interpolation=cv2.INTER_AREA), cv2.resize(plate, sz, interpolation=cv2.INTER_AREA), cv2.resize(lab, sz, interpolation=cv2.INTER_NEAREST)
    if WHICH == 'preview':
        fr = render(s_, p_, l_, 8*FPS, [2.6], None, None, None, frames_only=[0, 40, 60, 80, 120, 180, 239], sq=4096*sc*1080/MW)
        cv2.imwrite(f'{W}/qa/vtx_float_preview.jpg', np.hstack([cv2.resize(f, (360, 640)) for f in fr]))
    else:
        render(s_, p_, l_, 8*FPS, [2.6], None, f'{OUT}/spotify/vtx_spotify_9x16.mp4', None, sq=4096*sc*1080/MW)
elif WHICH == 'a34':
    TOP = 720; AW, AH = 2048, 2732
    cut = lambda im, interp: cv2.resize(im[TOP:TOP+4096], (AW, AH), interpolation=interp)
    s_, p_, l_ = cut(master, cv2.INTER_AREA), cut(plate, cv2.INTER_AREA), cut(lab, cv2.INTER_NEAREST)
    word = cv2.imread(f'{W}/vtx/vortex_word_rgba.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    lay = np.zeros((4096, MW, 4), np.float32); prev = None
    for (bx, by, bw, bh), y0, src in [((591, 254, 2917, 729), 700, None), ((2499, 3472, 1328, 379), 'after', word)]:
        part = tl[by:by+bh, bx:bx+bw].astype(np.float32) if src is None else src
        crop = cv2.resize(part, (int(bw*sc), int(bh*sc)), interpolation=cv2.INTER_AREA)
        x0 = int(round((bx+bw/2)*sc+M[0, 2]-crop.shape[1]/2)); y0 = prev+12-int(60*sc) if y0 == 'after' else y0
        lay[y0:y0+crop.shape[0], x0:x0+crop.shape[1]] = crop; prev = y0+crop.shape[0]
    lay = cv2.resize(lay, (AW, AH), interpolation=cv2.INTER_AREA); la = lay[..., 3:4]/255; lc = lay[..., :3]
    grad = (1-0.55*np.clip((np.arange(AH)/AH-0.60)/0.40, 0, 1)**1.4)[:, None, None]
    render(s_, p_, l_, 15*FPS, APPLE_SWEEPS, lambda f: lc*la + f*grad*(1-la), f'{OUT}/apple/vtx_apple_3x4.mp4', '90M', sq=4096*sc*AW/MW)
else:  # a11 — the cover square; plate and labels mapped from master into square coordinates
    S1 = 3840
    cover = cv2.imread(f'{W}/vtx/sq_text_4k.png').astype(np.float32); cl = cv2.imread(f'{W}/vtx/clean_4k.png').astype(np.float32)
    al = tl[..., 3:4].astype(np.float32)/255
    src = cv2.resize((cl*al + cover*(1-al)).astype(np.uint8), (S1, S1), interpolation=cv2.INTER_AREA)
    Msq = (np.vstack([M, [0, 0, 1]]) @ np.diag([4096/S1, 4096/S1, 1]))[:2].astype(np.float32)   # square(3840) -> master
    p_ = cv2.warpAffine(plate, Msq, (S1, S1), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    l_ = cv2.warpAffine(lab, Msq, (S1, S1), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    ta = cv2.resize(tl, (S1, S1), interpolation=cv2.INTER_AREA).astype(np.float32); a1 = ta[..., 3:4]/255; tc = ta[..., :3]
    render(src, p_, l_, 15*FPS, APPLE_SWEEPS, lambda f: tc*a1 + f*(1-a1), f'{OUT}/apple/vtx_apple_1x1.mp4', '90M', sq=S1,
           edge_block=cv2.dilate(a1[..., 0], np.ones((15, 15), np.uint8)))
