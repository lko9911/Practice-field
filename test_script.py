#시리즈 1.
import pandas as pd
sr = pd.Series([17000, 18000, 1000, 5000],
index = ["피자", "치킨", "콜라", "맥주"])

print('시리즈 출력 :')
print('-'*15)
print(sr)

#시리즈 2. 
print('시리즈의 값 : {}'.format(sr.values))
print('시리즈의 인덱스 : {}'.format(sr.index))

#데이터 프레임1.
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns) #pd.DataFrame()은 pandas 라이브러리에서 데이터프레임 객체를 생성하기 위한 함수

print('데이터프레임 출력 :')
print('-'*18)
print(df)

#데이터 프레임2. 
print('테이터프레임의 인덱스 : {}'.format(df.index))
print('데이터프레임의 열이름: {}'.format(df.columns))
print('데이터프레임의 값:')
print('-'*18)
print(df.values) #df.valuse 데이터프레임의 값만 반환

#데이터프레임의 생성3.1 리스트(행 단위 데이터 정의)
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]

df = pd.DataFrame(data)
print(df)

df = pd.DataFrame(data, columns=['학번', '이름', '점수']) #열 이름 별도 지정
print(df)

#데이터프레임의 생성3.2 딕셔너리(열 단위 데이터 정의의)
data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
    }

df = pd.DataFrame(data)
print(df)

#데이터프레임 조회4.
print(df.head(3)) #행 기준 상위 n개 행(0번 인덱스~)
print(df.tail(3))
print(df['학번'])

#외부 데이터 읽기
df = pd.read_csv('example.csv')
print(df)
print(df.index)
