# 거리 변환으로 전신 스켈레톤 찾기 (distanceTrans.py)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 읽어서 바이너리 스케일로 변환
img = cv2.imread('dataset/segment/masks_convert/MIDAS_forceps_285.png', cv2.IMREAD_GRAYSCALE)
_, biimg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# 거리 변환 ---①
dst = cv2.distanceTransform(biimg, cv2.DIST_L2, 5)
# 거리 값을 0 ~ 255 범위로 정규화 ---②
dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)
# 거리 값에 쓰레시홀드로 완전한 뼈대 찾기 ---③
skeleton = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                 cv2.THRESH_BINARY, 7, -3)
# 결과 출력
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.imshow(img)
plt.subplot(3,1,2)
plt.imshow(dst)
plt.subplot(3,1,3)
plt.imshow(skeleton)
plt.show()


# cv2.imshow('origin', img)
# cv2.imshow('dist', dst)
# cv2.imshow('skel', skeleton)
# cv2.waitKey(0)
# cv2.destroyAllWindows()