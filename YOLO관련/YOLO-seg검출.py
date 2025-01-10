import cv2
import numpy as np
from ultralytics import YOLO
import torch

# CUDA 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 이미지 경로 및 YOLO 모델 로드
img_path = "images/sample.jpg"
yolo_model = YOLO("best.pt")

# 이미지 읽기
img = cv2.imread(img_path)

# YOLO 예측
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=True, show=True)

# 결과 이미지 추출
output_img = results.ims[0]

# 이미지 크기 20%로 축소
height, width = output_img.shape[:2]
new_width = int(width * 0.2)
new_height = int(height * 0.2)
output_resized = cv2.resize(output_img, (new_width, new_height))

# 축소된 이미지 출력
cv2.imshow("Resized Image", output_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
