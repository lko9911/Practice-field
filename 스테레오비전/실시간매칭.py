import cv2
import numpy as np

# 스테레오 매칭 알고리즘 초기화
stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)

# 비디오 스트림 초기화
cap = cv2.VideoCapture(0)  # 스테레오 카메라 인덱스

# ORB 생성
orb = cv2.ORB_create()

# 비디오 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 읽을 수 없습니다.")
        break

    # 이미지를 좌우로 분리 (가정: 병합된 이미지)
    h, w, _ = frame.shape
    half_width = w // 2
    img_left = frame[:, :half_width]
    img_right = frame[:, half_width:]

    # 그레이스케일로 변환
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 스테레오 매칭으로 disparity 계산
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # 시각화를 위한 정규화
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    # 컬러맵 적용
    disparity_colormap = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    # 매칭 포인트 계산 및 연결 선 그리기
    keypoints1, descriptors1 = orb.detectAndCompute(gray_left, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_right, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 매칭을 거리순으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    # 매칭 결과 그리기 (상위 30개 매칭)
    match_img = cv2.drawMatches(img_left, keypoints1, img_right, keypoints2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 매칭 선 추가
    combined = np.hstack((img_left, img_right))
    line_spacing = 20  # 매칭 선 간격
    for i in range(0, h, line_spacing):
        cv2.line(combined, (0, i), (w, i), (0, 255, 0), 1)

    # 결과 디스플레이
    cv2.imshow('Stereo View with Matching Lines', combined)
    cv2.imshow('Disparity Map (Gray)', disparity_normalized)
    cv2.imshow('Disparity Map (Colormap)', disparity_colormap)
    cv2.imshow('Feature Matches', match_img)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q' 키로 프로그램 종료
        break
    elif key == ord('s'):  # 's' 키로 사진 저장
        cv2.imwrite('left_image.png', img_left)
        cv2.imwrite('right_image.png', img_right)
        cv2.imwrite('disparity_gray.png', disparity_normalized)
        cv2.imwrite('disparity_colormap.png', disparity_colormap)
        print("이미지를 저장했습니다.")

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
