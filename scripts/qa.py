"""qa.py <video> <sheet.jpg> : spec line, loop-seam jump vs typical frame step, contact sheet."""
import cv2, numpy as np, sys, json, subprocess
v = sys.argv[1]
pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,color_primaries:format=duration,bit_rate', '-of', 'json', v], capture_output=True, text=True).stdout)
vs = [s for s in pr['streams'] if s['codec_type'] == 'video'][0]; aud = [s for s in pr['streams'] if s['codec_type'] == 'audio']
cap = cv2.VideoCapture(v); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(cv2.resize(f, (360, int(360*f.shape[0]/f.shape[1]))).astype(np.float32))
d = [np.abs(fr[i+1]-fr[i]).mean() for i in range(len(fr)-1)]
print(f"{v.split('/')[-1]}: {vs['width']}x{vs['height']} {vs['codec_name']} {vs['pix_fmt']} {vs['r_frame_rate']} frames={len(fr)} dur={float(pr['format']['duration']):.2f}s "
      f"{int(pr['format']['bit_rate'])/1e6:.1f}Mbps audio={'YES' if aud else 'none'} prim={vs.get('color_primaries')} | step mean {np.mean(d):.2f} max {np.max(d):.2f} | loop seam {np.abs(fr[0]-fr[-1]).mean():.2f}")
idx = np.linspace(0, len(fr)-1, 6).astype(int)
cv2.imwrite(sys.argv[2], np.hstack([fr[i] for i in idx]).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
