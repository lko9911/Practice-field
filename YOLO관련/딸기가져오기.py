import os
import cv2
from ultralytics import YOLO
import torch

# CUDA 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# YOLO 모델 로드
yolo_model = YOLO("best.pt")

# 데이터 경로 설정
base_dir = "Dataset"
output_dir_healthy = "cropped_objects/healthy_strawberry"
output_dir_infected = "cropped_objects/infected_strawberry"

# 결과 저장 디렉토리 생성
os.makedirs(output_dir_healthy, exist_ok=True)
os.makedirs(output_dir_infected, exist_ok=True)

# 디렉토리 순회
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if not os.path.isdir(category_path):
        continue

    # 저장 디렉토리 설정
    if category == "healthy_strawberry":
        category_output_dir = output_dir_healthy
    elif category == "infected_strawberry":
        category_output_dir = output_dir_infected
    else:
        print(f"Unknown category: {category}, skipping...")
        continue

    # 이미지 처리
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        # 이미지 읽기
        img = cv2.imread(image_path)

        # YOLO 예측
        results = yolo_model.predict(image_path, imgsz=640, conf=0.5, save=False, show=False)
        detections = results[0].boxes  # YOLO 검출 결과

        # 검출된 객체 잘라내기 및 저장
        for idx, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cropped_object = img[y1:y2, x1:x2]

            # 저장 경로 생성
            save_path = os.path.join(category_output_dir, f"{os.path.splitext(image_name)[0]}_object_{idx}.jpg")
            cv2.imwrite(save_path, cropped_object)
            print(f"Saved: {save_path}")

print("All images have been processed.")
