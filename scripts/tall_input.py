"""tall_input.py <W> <key>

Gemini outpaint input for the 9:16 master: the clean square at 1152 px, placed so that
648/4096 of the height sits above it (the rest below), with heavily blurred stretches of its own
top and bottom rows as placeholders. Gemini is told to replace the blurred bands with real painting.
(Feeding the bare square instead makes Gemini shrink the art into a picture-in-picture.)
"""
import cv2, numpy as np, sys

W, k = sys.argv[1:3]
s = cv2.resize(cv2.imread(f'{W}/{k}/clean_4k.png'), (1152, 1152), interpolation=cv2.INTER_AREA)
H = 2048; y0 = round(648/4096*H)
can = np.zeros((H, 1152, 3), np.uint8); can[y0:y0+1152] = s
can[:y0] = cv2.GaussianBlur(cv2.resize(s[:40], (1152, y0)), (0, 0), 25)
can[y0+1152:] = cv2.GaussianBlur(cv2.resize(s[-40:], (1152, H-y0-1152)), (0, 0), 25)
cv2.imwrite(f'{W}/{k}/tall_input.png', can)
print('->', f'{W}/{k}/tall_input.png')
