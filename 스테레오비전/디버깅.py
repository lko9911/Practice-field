import cv2
import numpy as np

# 예시 카메라 매트릭스 (왼쪽/오른쪽 카메라)
cameraMatrix1 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
cameraMatrix2 = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

# 왜곡 계수 (이 예시에서는 0으로 설정)
distCoeffs1 = np.zeros((5,))
distCoeffs2 = np.zeros((5,))

# 이미지 크기 (960, 1280으로 설정)
imageSize = (1280, 960)  # 예시 이미지 크기

# 회전 행렬 (두 카메라 사이의 회전)
R = np.array([[ 9.99775314e-01, -2.11971661e-02, -4.91799203e-05],
              [ 2.11971636e-02,  9.99775314e-01, -5.17500015e-05],
              [ 5.02658236e-05,  5.06958992e-05,  9.99999997e-01]])

# 변위 벡터 (두 카메라 사이의 상대적 위치)
T = np.array([[-6.26590558], [-3.25829152], [-0.00645554]])

# 스테레오 이미지 불러오기
img_left = cv2.imread('left/left_image_1.png')
img_right = cv2.imread('right/right_image_1.png')

# 이미지 크기 (h, w)
h, w = img_left.shape[:2]
assert (w, h) == imageSize, "이미지 크기가 설정한 크기와 일치하지 않습니다."

# stereoRectify 함수 실행
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T
)

# 정합된 이미지를 위한 맵을 생성
map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), 5)
map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), 5)

# 좌우 이미지를 정합
rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

# 정합된 이미지 시각화
cv2.imshow("Rectified Left Image", rectified_left)
cv2.imshow("Rectified Right Image", rectified_right)

cv2.waitKey(0)
cv2.destroyAllWindows()

