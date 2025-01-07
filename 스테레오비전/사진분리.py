import cv2
import numpy as np
import glob

# 이미지 파일 경로 (디렉토리에서 *.png 파일을 모두 찾음)
image_paths = glob.glob('images/*.jpg')  # 디렉토리 경로에 맞게 수정

# 각 이미지를 처리
for i, img_path in enumerate(image_paths):
    
    img = cv2.imread(img_path)

    # 이미지 크기 확인
    height, width, _ = img.shape

    # 이미지를 좌우로 나누기
    left_img = img[:, :width//2]  # 왼쪽 이미지
    right_img = img[:, width//2:]  # 오른쪽 이미지

    # 각 이미지를 별도로 저장
    left_image_path = f'left/left_image_{i+1}.png'
    right_image_path = f'right/right_image_{i+1}.png'

    cv2.imwrite(left_image_path, left_img)
    cv2.imwrite(right_image_path, right_img)

    # 이미지를 화면에 표시하여 확인
    cv2.imshow(f'left image {i+1}', left_img)
    cv2.imshow(f'right image {i+1}', right_img)

# 대기 시간
cv2.waitKey(1000)
cv2.destroyAllWindows()
