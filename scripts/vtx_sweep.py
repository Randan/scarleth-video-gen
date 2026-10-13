"""vtx_sweep.py <W> <outdir> <which: spotify|a34|a11|preview>

Vortex loop with NO geometry change (no zoom, no swirl, no warping): the still holds, and once per loop (twice in the 15 s Apple loops) a
quick flashlight-like beam passes behind the camera, so the glowing glass edges flare as the beam crosses
them. The beam is a soft diagonal band that travels across the whole frame in SWEEP seconds; before and
after it the frame is the untouched still, so frame 0 is the cover and the loop seam is invisible.
Lettering is a static overlay and never lights up."""
import cv2, numpy as np, json, subprocess, sys, os

W, OUT, WHICH = sys.argv[1:4]
FPS = 30
SWEEP = 1.1                      # seconds for the beam to cross the frame
ANGLE = np.radians(28)           # beam travel direction: left -> right, slightly downward
WIDTH = 0.085                    # beam half-width, fraction of the frame diagonal
APPLE_SWEEPS = [1.5, 9.0]        # 15 s Apple loops: two passes (one read as a still image for too long)
master = cv2.imread(f'{W}/vtx/master_916.png'); MH, MW = master.shape[:2]
reg = json.load(open(f'{W}/vtx/reg_master.json')); M = np.array(reg['M'], np.float32); sc = reg['scale']
tl = cv2.imread(f'{W}/vtx/text_layer.png', cv2.IMREAD_UNCHANGED)

def edge_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    e = np.clip((hsv[..., 2]-170)/60, 0, 1)*np.clip((100-hsv[..., 1])/70, 0, 1)
    return cv2.GaussianBlur(e, (0, 0), 1.2)

def enc(path, w, h, bitrate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    args = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'medium', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-threads', '6',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:ipratio=1.0:pbratio=1.0:keyint=600:min-keyint=600:scenecut=0', '-movflags', '+faststart']
    if bitrate is None: args += ['-crf', '4']
    else:  # near-static frames: force constant bitrate so the file stays inside Apple's 45-100 Mbps window
        i = args.index('-x264-params'); args[i+1] += ':nal-hrd=cbr:force-cfr=1'
        args += ['-b:v', bitrate, '-minrate', bitrate, '-maxrate', bitrate, '-bufsize', bitrate]
    return subprocess.Popen(args+[path], stdin=subprocess.PIPE)

def render(src, length, starts, post, path, bitrate, edge_block=None):
    """src: still (uint8) at output size; starts: seconds at which a beam enters; edge_block: alpha where edges must not flare."""
    H, Wd = src.shape[:2]
    E = edge_mask(src)
    if edge_block is not None: E *= (1-edge_block)
    glow = cv2.GaussianBlur(E, (0, 0), max(2, Wd/300))       # tight halo around the lit edges
    bloom = cv2.GaussianBlur(E, (0, 0), max(6, Wd/70))         # wide scatter of the beam in the glass
    bloom /= max(1e-6, float(np.percentile(bloom, 99.5)))
    yy, xx = np.mgrid[0:H, 0:Wd].astype(np.float32)
    D = np.hypot(H, Wd)
    proj = ((xx*np.cos(ANGLE) + yy*np.sin(ANGLE)) / D).astype(np.float32)
    lo, hi = float(proj.min())-3*WIDTH, float(proj.max())+3*WIDTH
    del xx, yy
    srcf = src.astype(np.float32); tint = np.array([255, 250, 235], np.float32)   # cold white (BGR)
    p = enc(path, Wd, H, bitrate)
    for i in range(length):
        t = next((i/FPS - st for st in starts if 0 <= i/FPS - st <= SWEEP), None)
        if t is not None:
            s = t/SWEEP; s = s*s*(3-2*s)                        # ease in/out, still quick
            pos = lo + (hi-lo)*s
            b = np.exp(-((proj-pos)/WIDTH)**2)
            f = srcf*(1 + 1.3*E*b)[..., None] + ((1.6*glow + 0.55*bloom)*b)[..., None]*tint
            f = f + (0.10*b)[..., None]*srcf                     # faint spill of the beam on the walls
        else:
            f = srcf
        if post is not None: f = post(f)
        p.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait(); print('->', path)

if WHICH in ('spotify', 'preview'):
    still = cv2.resize(master, (1080, 1920), interpolation=cv2.INTER_AREA)
    path = f'{OUT}/spotify/vtx_spotify_9x16.mp4' if WHICH == 'spotify' else f'{W}/qa/vtx_sweep_preview.mp4'
    render(still, 8*FPS, [2.6], None, path, None)
elif WHICH == 'a34':
    TOP = 720; AW, AH = 2048, 2732
    win = cv2.resize(master[TOP:TOP+4096], (AW, AH), interpolation=cv2.INTER_AREA)
    lay = np.zeros((4096, MW, 4), np.float32); prev = None
    word = cv2.imread(f'{W}/vtx/vortex_word_rgba.png', cv2.IMREAD_UNCHANGED).astype(np.float32)   # unblended against the clean art
    for (bx, by, bw, bh), y0, src in [((591, 254, 2917, 729), 700, None), ((2499, 3472, 1328, 379), 'after', word)]:
        part = tl[by:by+bh, bx:bx+bw].astype(np.float32) if src is None else src
        crop = cv2.resize(part, (int(bw*sc), int(bh*sc)), interpolation=cv2.INTER_AREA)
        x0 = int(round((bx+bw/2)*sc+M[0, 2]-crop.shape[1]/2)); y0 = prev+12-int(60*sc) if y0 == 'after' else y0
        lay[y0:y0+crop.shape[0], x0:x0+crop.shape[1]] = crop; prev = y0+crop.shape[0]
    lay = cv2.resize(lay, (AW, AH), interpolation=cv2.INTER_AREA); la = lay[..., 3:4]/255; lc = lay[..., :3]
    grad = (1-0.55*np.clip((np.arange(AH)/AH-0.60)/0.40, 0, 1)**1.4)[:, None, None]
    render(win, 15*FPS, APPLE_SWEEPS, lambda f: lc*la + f*grad*(1-la), f'{OUT}/apple/vtx_apple_3x4.mp4', '90M')
else:
    S1 = 3840
    cover = cv2.resize(cv2.imread(f'{W}/vtx/sq_text_4k.png'), (S1, S1), interpolation=cv2.INTER_AREA)
    ta = cv2.resize(tl[..., 3], (S1, S1), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    ta = cv2.dilate(ta, np.ones((15, 15), np.uint8))
    render(cover, 15*FPS, APPLE_SWEEPS, None, f'{OUT}/apple/vtx_apple_1x1.mp4', '70M', edge_block=ta)
