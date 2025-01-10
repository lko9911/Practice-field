import cv2
import numpy as np
from ultralytics import YOLO
import torch

# CUDA 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 이미지 경로 및 YOLO 모델 로드
img_path = "sample.jpg"
yolo_model = YOLO("best.pt")

# 이미지 읽기
img = cv2.imread(img_path)

# YOLO 예측
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=False)

# 마스크 추출
masks = results[0].masks

# 정규화된 좌표 xyn 추출
xyn = masks.xyn

# 원본 이미지 크기 (높이, 너비)
orig_height, orig_width = img.shape[:2]

# 마스크를 생성하여 영역 추출
for i, mask in enumerate(xyn):
    # xyn에서 각 세그먼트의 좌표를 원본 이미지 크기로 변환
    segment = mask * np.array([orig_width, orig_height])  # 정규화된 좌표를 원본 크기로 변환
    segment = segment.astype(np.int32)  # 정수형으로 변환하여 사용할 준비

    # 각 마스크를 polygon 형태로 만들어서, 마스크 영역을 표시
    mask_img = np.zeros((orig_height, orig_width), dtype=np.uint8)  # 마스크로 사용할 빈 이미지 생성
    cv2.fillPoly(mask_img, [segment], 255)  # 다각형을 채워서 마스크 영역 생성

    # 원본 이미지에서 마스크된 부분을 추출
    masked_img = cv2.bitwise_and(img, img, mask=mask_img)

    # 마스크된 이미지 HSV로 변환
    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

    # 빨간색 영역 추출: 빨간색은 H값이 0 또는 180도 근처에 있음
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 영역 마스크
    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # 두 범위의 빨간색을 합침

    # 하얀색 영역 추출: 하얀색은 S 값이 낮고 V 값이 높음
    lower_white = np.array([0, 0, 100])
    upper_white = np.array([200, 80, 255]) # 2번째 값 조정중중

    # 하얀색 영역 마스크
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white)

    # 빨간색 영역과 하얀색 영역을 추출
    red_area = cv2.bitwise_and(masked_img, masked_img, mask=mask_red)
    white_area = cv2.bitwise_and(masked_img, masked_img, mask=mask_white)

    # 결과 보여주기
    cv2.imshow(f"Masked Image {i+1}", masked_img)
    cv2.imshow(f"Red Area {i+1}", red_area)
    cv2.imshow(f"White Area {i+1}", white_area)

cv2.waitKey(0)
cv2.destroyAllWindows()
