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
    # Global Average Pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # 배치 차원 제거
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # ReLU 및 정규화
    return heatmap.numpy()


# Grad-CAM 히트맵 시각화 함수
def overlay_heatmap(img, heatmap, reverse_colormap=False):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    if reverse_colormap:
        heatmap = cv2.applyColorMap(255 - np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    else:
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
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


# 모델 불러오기
model = load_model('best_resnet50_model.keras')

# Grad-CAM 실행
img_path = "images/sample2.jpg"
img, img_tensor = preprocess_image(img_path)
predictions = model.predict(img_tensor)
class_idx = np.argmax(predictions[0])  # 가장 높은 점수의 클래스 선택

# 예측 클래스 확인
if predictions[0][class_idx] >= 0.5:
    predicted_class = "infected_strawberry"
else:
    predicted_class = "healthy_strawberry"
    
prediction_score = predictions[0][class_idx]

# Grad-CAM 계산
target_layer_name = "conv5_block3_out"  # ResNet50의 마지막 컨볼루션 레이어
heatmap = compute_gradcam(model, img_tensor, target_layer_name, class_idx)

# Grad-CAM 히트맵 생성
img_np = np.array(img) / 255.0  # [0, 1] 범위로 정규화된 원본 이미지
cam_image = overlay_heatmap(img_np, heatmap)

# 결과 출력
plt.figure(figsize=(10, 5))

# 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.title(f"Original Image\nPredicted: {predicted_class}\n")
plt.imshow(img_np)
plt.axis('off')

# Grad-CAM 출력
plt.subplot(1, 2, 2)
plt.title("Grad-CAM (Reversed)")
plt.imshow(cam_image)
plt.axis('off')

plt.show()

# 예측 클래스와 점수 출력
print(f"Predicted class: {predicted_class}")
