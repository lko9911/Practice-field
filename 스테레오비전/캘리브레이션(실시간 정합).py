import cv2
import numpy as np
import glob

# 체스보드 크기 설정 (체스보드의 칸 수, 예: 7x7)
pattern_size = (7, 7)
square_size = 0.03  

# 세계 좌표계에서의 3D 점 (체스보드의 각 점 위치)
obj_points = []  # 3D 점 (실제 좌표)
img_points_left = []  # 왼쪽 카메라의 2D 점
img_points_right = []  # 오른쪽 카메라의 2D 점

# 체스보드 이미지 파일 경로
left_images = glob.glob('left/*.png')  # 왼쪽 카메라 이미지 경로
right_images = glob.glob('right/*.png')  # 오른쪽 카메라 이미지 경로

# 3D 포인트 준비 (체스보드의 각 점에 대한 3D 좌표)
obj_p = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
obj_p[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
obj_p *= square_size  # 크기 적용

# 이미지 리사이즈 비율
resize_factor = 0.5  # 50% 크기로 축소 (예시)

# 각 이미지에 대해 체스보드 코너를 찾기
for left, right in zip(left_images, right_images):
    img_left = cv2.imread(left)
    img_right = cv2.imread(right)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

    if ret_left and ret_right:
        # 체스보드가 감지된 경우
        obj_points.append(obj_p)
        img_points_left.append(corners_left)
        img_points_right.append(corners_right)

        # 감지된 체스보드 코너를 이미지에 그려서 시각적으로 확인
        cv2.drawChessboardCorners(img_left, pattern_size, corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, pattern_size, corners_right, ret_right)

        # 두 이미지를 수평으로 이어서 정합 이미지 표시
        combined_img = np.hstack((img_left, img_right))

        # 리사이즈
        height, width = combined_img.shape[:2]
        new_dim = (int(width * resize_factor), int(height * resize_factor))
        resized_combined_img = cv2.resize(combined_img, new_dim)

        # 이미지 표시
        cv2.imshow('Stereo Calibration - Left & Right Images with Corners', resized_combined_img)
        cv2.waitKey(500)  # 500ms 동안 확인
        cv2.destroyAllWindows()

    else:
        print(f"체스보드가 감지되지 않았습니다: {left}, {right}")

# 스테레오 카메라 캘리브레이션
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right, 
    gray_left.shape[::-1], None, None, None, None, 
    criteria=criteria
)

# 캘리브레이션 결과 출력
if ret:
    print("캘리브레이션 성공!")
    print("Left Camera Matrix:\n", K1)
    print("Right Camera Matrix:\n", K2)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
else:
    print("캘리브레이션 실패!")

# 결과를 npz 파일로 저장
np.savez('stereo_calibration_result.npz',
         K1=K1, dist1=dist1,  
         K2=K2, dist2=dist2,  
         R=R, T=T,            
         E=E, F=F) 

# 저장된 npz 파일 불러오기
data = np.load('stereo_calibration_result.npz')

# 각 매트릭스를 접근하여 출력
K1 = data['K1']
dist1 = data['dist1']
K2 = data['K2']
dist2 = data['dist2']
R = data['R']
T = data['T']
E = data['E']
F = data['F']

# 출력 예시
print("Left Camera Matrix (K1):\n", K1)
print("Right Camera Matrix (K2):\n", K2)
print("Left Camera Distortion Coefficients (dist1):\n", dist1)
print("Right Camera Distortion Coefficients (dist2):\n", dist2)
print("Rotation Matrix (R):\n", R)
print("Translation Vector (T):\n", T)
print("Essential Matrix (E):\n", E)
print("Fundamental Matrix (F):\n", F)
