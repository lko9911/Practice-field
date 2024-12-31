# 라이브러리 가져오기 
from ultralytics import YOLO
import cv2
import torch

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "image/sample_1.png"

# YOLO 모델 로드
yolo_model = YOLO("model/tomato_yolo.pt")

# YOLO 모델로 객체 탐지 (신뢰도 조정)
results = yolo_model.predict(img_path, imgsz=640, conf=0.5, save=False, show=False)  # show=False로 설정 후, 별도로 처리

# 이미지 읽기
img = cv2.imread(img_path)

# 검출된 객체를 인덱싱하고 좌표를 출력
for i, result in enumerate(results[0].boxes):
    # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
    x1, y1, x2, y2 = map(int, result.xyxy[0])

    # 객체의 중심 좌표 계산
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # 인덱스 텍스트와 좌표 텍스트 설정
    label_text = f"Tomato {i+1}"
    
    # 텍스트 폰트 및 크기 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    text_color = (255, 255, 255)  # 흰색 텍스트
    background_color = (0, 107, 223)  # 녹색 배경

    # 텍스트 크기 계산
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)

    # 텍스트 배경 그리기 (텍스트의 배경 박스)
    cv2.rectangle(img, (x1, y1), (x1 + text_width + 10, y1 - text_height - 10), background_color, -1)

    # 텍스트 삽입 (좌표와 텍스트 위치 맞추기)
    cv2.putText(img, label_text, (x1 + 5, y1 - 5), font, font_scale, text_color, font_thickness)

    # 바운딩 박스 그리기
    cv2.rectangle(img, (x1, y1), (x2, y2), background_color, 2)  # 바운딩 박스 그리기

    # 인덱스와 좌표 출력
    print(f"Tomato {i+1}: Center coordinates = ({center_x}, {center_y})")

# 결과 이미지 표시
cv2.imshow("Tomato Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
