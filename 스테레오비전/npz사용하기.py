import cv2
import numpy as np

# 캘리브레이션 결과 불러오기
data = np.load('stereo_calibration_result.npz')
K1 = data['K1']
dist1 = data['dist1']
K2 = data['K2']
dist2 = data['dist2']
R = data['R']
T = data['T']

# 새로 찍은 스테레오 이미지 불러오기
img_left = cv2.imread('input/left_image_1.png')  # 왼쪽 이미지
img_right = cv2.imread('input/right_image_1.png')  # 오른쪽 이미지

# 리사이즈 비율 설정
resize_factor = 0.5  # 크기를 50%로 줄입니다.
h, w = img_left.shape[:2]
new_dim = (int(w * resize_factor), int(h * resize_factor))

# 이미지를 리사이즈
img_left_resized = cv2.resize(img_left, new_dim)
img_right_resized = cv2.resize(img_right, new_dim)

# 그레이스케일로 변환
gray_left = cv2.cvtColor(img_left_resized, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right_resized, cv2.COLOR_BGR2GRAY)

# 스테레오 정합
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, dist1, K2, dist2, new_dim, R, T)
map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, new_dim, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, new_dim, cv2.CV_32FC1)

rectified_left = cv2.remap(gray_left, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(gray_right, map2x, map2y, cv2.INTER_LINEAR)

# 스테레오 매칭 객체 생성
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*5,  # disparities 수 (16의 배수)
    blockSize=7,          # 매칭 블록 크기
    P1=8 * 3 * 7**2,      # Smoothness penalty 1
    P2=32 * 3 * 7**2,     # Smoothness penalty 2
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# 밀도 disparity 계산
disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0

# 시각화를 위한 정규화
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# 결과 출력
cv2.imshow('Rectified Left Image', rectified_left)
cv2.imshow('Rectified Right Image', rectified_right)
cv2.imshow('Disparity Map', disparity_normalized)

cv2.waitKey(0)
cv2.destroyAllWindows()
