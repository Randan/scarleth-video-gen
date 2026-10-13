"""register.py <square.png> <target.png> -> similarity transform square->target (json)"""
import cv2, numpy as np, sys, json
a = cv2.imread(sys.argv[1]); b = cv2.imread(sys.argv[2])
s = 1024/a.shape[1]; ag = cv2.cvtColor(cv2.resize(a, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
sb = 1024/b.shape[1]; bg = cv2.cvtColor(cv2.resize(b, None, fx=sb, fy=sb), cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(6000)
k1, d1 = sift.detectAndCompute(ag, None); k2, d2 = sift.detectAndCompute(bg, None)
m = cv2.BFMatcher().knnMatch(d1, d2, k=2); good = [x for x, y in m if x.distance < 0.72*y.distance]
p1 = np.float32([k1[g.queryIdx].pt for g in good])/s; p2 = np.float32([k2[g.trainIdx].pt for g in good])/sb
M, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=6)
print(json.dumps({'M': M.tolist(), 'scale': round(float(np.hypot(M[0, 0], M[1, 0])), 4), 'tx': round(float(M[0, 2]), 1),
                  'ty': round(float(M[1, 2]), 1), 'matches': len(good), 'inliers': int(inl.sum())}))
