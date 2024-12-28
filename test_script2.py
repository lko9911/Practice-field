import numpy as np
vec = np.array([1, 2, 3, 4, 5]) # 1차원 배열 / 리스트 => ndarray 변환
print(vec)

mat = np.array([[10, 20, 30], [60, 70, 80]]) #2차원 배열
print(mat)

# 배열의 타입
print('vec의 타입 :',type(vec))
print('mat의 타입 :',type(mat))

print('vec의 축의 개수 :',vec.ndim) # 축의 개수 출력 / ndim = 축의 개수
print('vec의 크기(shape) :',vec.shape) # 배열 크기 출력

print('mat의 축의 개수 :',mat.ndim) # 2개의 행에 3개의 열열
print('mat의 크기(shape) :',mat.shape) # (2, 3) 첫 번째 축(행): 2개의 요소, 두 번째 축(열): 각 행에 3개 요소소

#0인 2x3 배열 생성
zero_mat = np.zeros((2,3)) #빈 행렬을 만들고 , 모든 방 0으로 초기화 / 딥러닝(데이터 분석) 초기 값 0 설정
print(zero_mat)

#1인 2x3 배열 생성.
one_mat = np.ones((2,3))
print(one_mat)

# 값이 특정 상수인 배열 생성. 사용자가 지정한 값 7
same_value_mat = np.full((2,2), 7)
print(same_value_mat)

# 2차원 배열, 단위행렬 생성 
eye_mat = np.eye(3)
print(eye_mat)

# 임의의 값으로 채워진 배열 생성
random_mat = np.random.random((2,2)) #np.random.random() 0과 1 사이의 임의의 실수로 채워진 배열 생성 함수
print(random_mat)

#np.arange()
range_vec = np.arange(10) # 0~9까지 값 
print(range_vec)

n = 2
range_n_step_vec = np.arange(1, 10, n) # 1부터 9까지 +2씩 
print(range_n_step_vec)

#np.reshape()
reshape_mat = np.array(np.arange(30).reshape((5, 6))) #.reshape()는 배열 크기(shape) 변경 함수
print(reshape_mat) #가능한 shape: (2, 3, 5), (5, 6), (3, 10)..

#Numpy 슬라이싱
mat = np.array([[1, 2, 3],[4, 5, 6]])
print(mat)

slicing_mat = mat[0, :] # 0: 첫 번째 행의 인덱스, : 열 전체
print(slicing_mat)

slicing_mat = mat[1:, 1] # : 행 전체, 1: 열의 두 번째 값(인덱스 1)
print(slicing_mat)

#integer indexing
mat = np.array([[1, 2], [4, 5], [7, 8]])
print(mat)

#특정 원소 추출출
print(mat[1, 0]) #mat[1, 0] 1: 1번행 (0, 1, 2번 행에서) 0: 0번 인덱스(0, 1, 2번 인덱스에서서)

#특정 원소 => 새로운 배열 생성 
indexing_mat = mat[[2, 1], [0, 1]]
print(indexing_mat)

#Numpy 연산 
x = np.array([1, 2, 3])
y =np.array([4, 5, 6])

result = x + y
print(result)

result = x - y
print(result)

result = result * x # * 요소별 곱 
print(result)

result = result / x
print(result)

#두 행렬의 곱셈 = 행렬 곱(내적)
mat1 = np.array([[1,2], [3, 4]])
mat2 = np.array([[5, 6],[7, 8]])
mat3 = np.dot(mat1, mat2)
print(mat3)
