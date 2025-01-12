import cv2
import numpy as np
from ultralytics import YOLO
import torch
import glob

# CUDA 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# YOLO 모델 로드
yolo_model = YOLO("best.pt")

# 이미지 폴더 경로 설정 (슬래시 변경)
folder_path = "G:/내 드라이브/Strawberry_Yolo_Hsv/infected_strawberry"  # 개인 폴더 경로로 재수정 필요
image_paths = glob.glob(f"{folder_path}/*.jpg")  # 폴더 내 모든 JPG 파일 가져오기

# 각 이미지를 처리
for img_path in image_paths:
    # 이미지 읽기
    img = cv2.imread(img_path)

    # YOLO 예측
    results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=False)

    # 마스크 추출
    masks = results[0].masks

    # 마스크가 None인지 확인
    if masks is None:
        print(f"No masks detected for image: {img_path}")
        continue  # 다음 이미지로 넘어감

    # 정규화된 좌표 xyn 추출
    xyn = masks.xyn

    # 원본 이미지 크기 (높이, 너비)
    orig_height, orig_width = img.shape[:2]

    # 각 마스크를 처리하여 흰색 영역 추출
    for i, mask in enumerate(xyn):
        # xyn에서 각 세그먼트의 좌표를 원본 이미지 크기로 변환
        segment = mask * np.array([orig_width, orig_height])  # 정규화된 좌표를 원본 크기로 변환
        segment = segment.astype(np.int32)  # 정수형으로 변환

        # 마스크 생성 및 영역 표시
        mask_img = np.zeros((orig_height, orig_width), dtype=np.uint8)
        cv2.fillPoly(mask_img, [segment], 255)

        # 원본 이미지에서 마스크된 부분 추출
        masked_img = cv2.bitwise_and(img, img, mask=mask_img)

        # 마스크된 이미지 HSV로 변환
        hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

        # 하얀색 영역 추출: HSV 범위 설정
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([200, 95, 255])
        mask_white = cv2.inRange(hsv_img, lower_white, upper_white)

        # 흰색 영역 추출 결과 생성
        white_area = cv2.bitwise_and(masked_img, masked_img, mask=mask_white)

        # 결과 저장 또는 표시 (선택 사항)
        output_path = f"output_folder/white_area_{i}_{img_path.split('/')[-1]}"
        cv2.imwrite(output_path, white_area)  # 결과 저장

        # 화면에 결과 표시 (1초 간격으로 출력)
        cv2.imshow(f"Masked Image {i+1}", masked_img)       # 마스크된 부분 이미지 출력
        cv2.imshow(f"White Area {i+1}", white_area)         # 흰색 영역 추출 결과 이미지 출력
        cv2.waitKey(1000)  # 1초 대기 (1000ms)

cv2.destroyAllWindows()
