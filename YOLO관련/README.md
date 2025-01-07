# 🔖 Roboflow 데이터로 YOLOv11n 재학습 


- <h4>참고 사이트 : https://www.youtube.com/watch?v=RaY_9i6XOos
- <h4>재학습 결과 링크 (구글 드라이브) : https://drive.google.com/drive/folders/1nQugI_fbd-Wn67JN7mOVZZhCyhRNzGU5?usp=drive_link

---------------------------------------

<h3> 0. YOLOv11n 재학습.ipynb 파일을 구글 Colab에 올립니다. <br><br>

![스크린샷 2025-01-07 092325](https://github.com/user-attachments/assets/0ab0a007-7295-49f2-88da-6f5ad4824bbd)

<h4>이 작업은 이미 ipynb 파일에 포함되어 있습니다.

---------------------------------------

<h3> 1. YOLOv11n 재학습.ipynb 파일을 구글 Colab에 올립니다. <br><br>
  
![스크린샷 2025-01-07 092134](https://github.com/user-attachments/assets/14c0d520-38f4-454e-9928-f8b4178ac6bd)

---------------------------------------

<h3> 2. GPU 설정 (런타임 유형 바꾸기) <br><br>
  
![스크린샷 2025-01-07 092646](https://github.com/user-attachments/assets/8e3c41c4-c0a9-44df-9f5d-79f4ede57517)


<h4>저는 구글 프로 버전을 사용하기 때문에 A100 GPU를 사용하였고, 다른분들은 사용할수 있는 GPU로 설정해요 !

---------------------------------------

<h3> 3. 설정 바꾸기 <br><br>

<pre><code>results = model.train(data='/content/Strawberry_segmentation-2/data.yaml', 
  epochs=100, imgsz=640, batch=8)</code></pre>
<h4>여기서 epochs는 학습 횟수를 의미하는데 A100 GPU가 아니면 시간이 매우 오래 걸려서 10 ~ 20으로 설정을 바꾸시는걸 추천 드려요 ! <br><br> 다음으로 전체 코드를 실행시켜주면 학습합니다. 아래의 구글 마운틴 코드는 재학습 결과를 따로 구글 드라이브에 저장하는 코드입니다.<br><br>

<pre><code>from google.colab import drive
drive.mount('/content/drive')</code></pre>
<pre><code>import os
import shutil
from google.colab import drive

# Google 드라이브 마운트
drive.mount('/content/drive')

# 원본 경로
source_path = '/content/runs'

# 목적지 경로
destination_path = '/content/drive/MyDrive/Strawberry-seg-result'

# 목적지 경로가 없으면 생성
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

# source_path 경로에 있는 모든 파일과 폴더를 이동
for filename in os.listdir(source_path):
    file_path = os.path.join(source_path, filename)
    if os.path.isfile(file_path) or os.path.isdir(file_path):
        shutil.move(file_path, destination_path)('/content/drive')</code></pre>

<h4> runs/train/weights 에 있는 pt 파일이 최종 YOLO 파일이고 보통은 best.pt를 사용합니다.
