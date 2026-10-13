"""logo_rgba.py <W> <key>

The band logo alone (no album title), as a tight RGBA cut from the 4K cover -> <key>/logo_rgba.png.
  vtx       the thick Vortex lettering from the band's t-shirt PSD (stage only)
  kpd       cover pixels, alpha from the band's vector letters registered onto the cover (see below)
  bts       the band's own file logos/logos_old/logo.psd, layer "Group 8" (the same orange chrome lettering,
            clean alpha) -> work/logos/logo_old_L2.png. The cover cut had holes and a clipped bottom.
  slv       letters only, no ornament: the band's poster/shared/silver-lining-scarleth-logo.psd -> work/logos/sl_logo.png
"""
import cv2, numpy as np, sys

W, k = sys.argv[1:3]

if k == 'bts':
    crop = cv2.imread(f'{W}/logos/logo_old_L2.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
elif k == 'slv':
    # Letters only, no ornament (Andrew, 2026-10-12): the band's own Silver Lining lettering file, clean alpha.
    crop = cv2.imread(f'{W}/logos/sl_logo.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
elif k == 'kpd':
    # Pixels from the cover, alpha = the band's vector letters (logo_transparent.psd layer 3 -> logo2025_L2.png)
    # registered onto the cover with ECC (kpd/letters_alpha.npy), grown by ~6 px for the letters' own dark
    # outline. The plain cover cut carried a dark teal halo of the cover's sky (read as a blue glow).
    cov = cv2.imread(f'{W}/kpd/sq_text_4k.png').astype(np.float32)
    L0 = np.load(f'{W}/kpd/letters_alpha.npy')
    A = cv2.erode(cv2.dilate(L0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))), np.ones((1, 1), np.uint8))
    a = cv2.GaussianBlur(np.clip(A, 0, 1), (0, 0), 1.0)
    # A thicker black outline (Viktor, stage v6: "like on the streaming videos"): a near-black stroke ~15 px wide
    # at 4096 scale (~6 px on the 1080p stage), replacing the cover's thin rim and the sky specks caught in it.
    S = cv2.GaussianBlur(np.clip(cv2.dilate(L0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), 0, 1), (0, 0), 1.5)
    black = np.full_like(cov, 12)
    col = cov*a[..., None] + black*(1-a[..., None])
    crop = np.dstack([col, np.maximum(a, S)*255])
elif k == 'vtx':
    # Stage only: the thick icy Vortex lettering Andrew made for the merch (Viktor asked for a thicker logo):
    # t-shirt/2_vortex/back.psd, group "Group 1" -> work/logos/vtx_thick.png. The canvases keep the cover's logo.
    crop = cv2.imread(f'{W}/logos/vtx_thick.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
elif k == 'vtx_cover':
    tl = cv2.imread(f'{W}/{k}/text_layer.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    x, y, w, h = 591, 254, 2917, 729
    crop = tl[y:y+h, x:x+w].copy()
a = crop[..., 3]
ys, xs = np.where(a > 8)
crop = crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
cv2.imwrite(f'{W}/{k}/logo_rgba.png', np.clip(crop, 0, 255).astype(np.uint8))
print(k, 'logo', crop.shape[1], 'x', crop.shape[0])
