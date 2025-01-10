import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Grad-CAM 계산 함수
def compute_gradcam(model, img_tensor, target_layer_name, class_idx):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    # Conv 레이어의 그라디언트 계산
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # 배치 차원 제거
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # ReLU 및 정규화
    return heatmap.numpy()


# Grad-CAM 히트맵 시각화 함수
def overlay_heatmap(img, heatmap, reverse_colormap=False):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    if reverse_colormap:
        heatmap = cv2.applyColorMap(255 - heatmap, cv2.COLORMAP_JET)
    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed_img = heatmap + np.float32(img)
    overlayed_img = overlayed_img / np.max(overlayed_img)
    return np.uint8(255 * overlayed_img)


# 이미지 전처리 함수
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(128, 128))  # 모델 입력 크기에 맞게 조정
    img_array = img_to_array(img) / 255.0  # 정규화
    img_tensor = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img, img_tensor


# 이미지 회전 함수 (90도)
def rotate_image(img, angle):
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_img


# 모델 불러오기
model_path = 'best_resnet50_model.keras'  # 모델 파일 경로
model = load_model(model_path)

# Grad-CAM 실행
img_path = "images/infected_1.jpg"  # 테스트 이미지 경로
img, img_tensor = preprocess_image(img_path)

# 이미지를 회전
img_np = np.array(img) / 255.0  # 원본 이미지를 [0, 1]로 정규화
rotated_img_np = rotate_image((img_np * 255).astype(np.uint8), 180) / 255.0

# 회전된 이미지를 텐서로 변환
rotated_img_tensor = np.expand_dims(rotated_img_np, axis=0)

# 예측 및 Grad-CAM
predictions = model.predict(rotated_img_tensor)
class_idx = np.argmax(predictions[0])  # 가장 높은 점수의 클래스 선택

# 예측 클래스 및 점수 확인
predicted_class = "infected_strawberry" if predictions[0][class_idx] >= 0.5 else "healthy_strawberry"
prediction_score = predictions[0][class_idx]

# Grad-CAM 계산
target_layer_name = "conv5_block3_out"  # ResNet50의 마지막 컨볼루션 레이어
heatmap = compute_gradcam(model, rotated_img_tensor, target_layer_name, class_idx)

# Grad-CAM 히트맵 생성
cam_image = overlay_heatmap(rotated_img_np, heatmap)

# 결과 출력
plt.figure(figsize=(10, 5))

# 회전된 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.title(f"Rotated Image\nPredicted: {predicted_class}\nScore: {prediction_score:.2f}")
plt.imshow(rotated_img_np)
plt.axis('off')

# Grad-CAM 출력
plt.subplot(1, 2, 2)
plt.title("Grad-CAM Visualization (Rotated)")
plt.imshow(cam_image)
plt.axis('off')

plt.show()

# 예측 클래스와 점수 출력
print(f"Predicted class: {predicted_class}, Score: {prediction_score:.2f}")
