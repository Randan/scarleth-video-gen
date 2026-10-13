"""prepare_sources.py <W>

Copies the source covers into <W>/<key>/ as PNG:
  orig.png       the released cover, with lettering (kpd, bts, slv, vtx)
  clean_src.png  a textless original, where one exists (slv: sticker art, vtx: sticker art)
For Vortex the textless sticker art is wider than the cover, so it is registered onto the cover
square (SIFT + RANSAC similarity) and written as clean_4k.png directly; sq_text_4k.png is the
cover upsized to 4096 with Lanczos, since Vortex is already 3000 px and needs no re-render.
"""
import cv2, numpy as np, os, shutil, subprocess, sys

W = sys.argv[1]
DRIVE = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-andrewrandan@gmail.com/Мій диск/Scarleth')
SOURCES = {
    'kpd': {'orig': 'covers/Коли Прийде День.png'},
    'bts': {'orig': 'covers/Break The Silence.jpg'},
    'slv': {'orig': 't-shirt/3_silver_lining/обложка альбома.jpg',
            'clean_src': 'stickers/QR визитка техзадание/фон/cover.tif'},
    'vtx': {'orig': 'covers/Vortex.jpg',
            'clean_src': 'stickers/QR визитка техзадание/фон/cover clean.jpg'},
}

for key, files in SOURCES.items():
    os.makedirs(f'{W}/{key}', exist_ok=True)
    for name, rel in files.items():
        src, dst = f'{DRIVE}/{rel}', f'{W}/{key}/{name}.png'
        if src.lower().endswith('.png'):
            shutil.copy(src, dst)
        else:
            subprocess.run(['sips', '-s', 'format', 'png', src, '--out', dst], check=True, capture_output=True)
        print(key, name, '<-', rel)

o = cv2.imread(f'{W}/vtx/orig.png'); c = cv2.imread(f'{W}/vtx/clean_src.png')
s = 0.35
og = cv2.cvtColor(cv2.resize(o, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
cg = cv2.cvtColor(cv2.resize(c, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(8000)
k1, d1 = sift.detectAndCompute(og, None); k2, d2 = sift.detectAndCompute(cg, None)
good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.7*b.distance]
src = np.float32([k1[g.queryIdx].pt for g in good])/s; dst = np.float32([k2[g.trainIdx].pt for g in good])/s
M, inl = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=4)
M4 = M*(4096/o.shape[1])
cv2.imwrite(f'{W}/vtx/clean_4k.png', cv2.warpAffine(c, M4, (4096, 4096), flags=cv2.INTER_LANCZOS4))
cv2.imwrite(f'{W}/vtx/sq_text_4k.png', cv2.resize(o, (4096, 4096), interpolation=cv2.INTER_LANCZOS4))
print('vtx clean registered, inliers', int(inl.sum()), 'of', len(good))
