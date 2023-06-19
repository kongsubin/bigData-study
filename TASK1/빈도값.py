# 문제 : AGE 컬럼을 소숫점 첫번째 자리에서 반올림하고, 가장 많은 비중을 차지하는 AGE값과 
# 그 개수를 차례대로 출력, 즉 AGE칼람의 최빈값과 그 개수 출력 

import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")


### 첫번째 방법 
# AGE 컬럼 확인
print(data['AGE'])

# AGE 컬럼을 소숫점 첫번째 자리에서 반올림
print(round(data['AGE'], 0))
data2 = round(data['AGE'], 0)

# 데이터 프레임으로 타입변경
data2 = pd.DataFrame(data2)

# AGE 컬럼으로 그룹화하고, 그룹별 AGE 컬럼의 개수 세기
print(data2.groupby(['AGE'])['AGE'].count())

# 그룹화 결과 저장
data3 = data2.groupby(['AGE'])['AGE'].count()
print(data3)

# 데이터 프레임으로 타입변경
data3 = pd.DataFrame(data3)
print(type(data3))

# 칼럼 이름 확인
print(data3.columns)
# 칼람명 변경
data3.columns = ['COUNT']
print(data3.head(3))

# AGE라는 인덱스를 버리지 않고 그래도 칼럼으로 사용
data3.reset_index(drop=False, inplace=True)
print(data3.head(3))

# 최빈값을 찾기 위해 내림차순
print(data3.sort_values(by = 'COUNT', ascending=False))
data3.sort_values(by = 'COUNT', ascending=False, inplace=True)

# AGE칼람의 최빈값과 그 개수 출력 
print(data3.iloc[0,0], data3.iloc[0,1])


### 두번째 방법 
# 최빈값의 기능이 구현된 mode()함수 가져오기
from scipy.stats import mode

# data2의 최빈값과 개수 구하기
# print(mode(data2))
# # 최빈값만
# print(mode(data2)[0])
# # 개수만 
# print(mode(data2)[1])
# # 정수형 변환
# print(int(mode(data2)[0]), int(mode(data2)[1]))