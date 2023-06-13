# 문제 : MEDV 칼람 값들의 최솟값 0, 최댓값 1의 범위로 변환 후 0.5보다 큰 값을 가지는 레코드 수 반환

import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# 최소최대 척도로 변환
data_minmax = scaler.fit_transform(data)

# 데이터 타입 확인
print(type(data_minmax))

# 데이터프레임으로 변경 
data_minmax = pd.DataFrame(data_minmax, columns=data.columns) 
# columns=data.columns : 컬럼명을 그대로 사용하고자함 
print(data_minmax.head(3))

# 기초통계량 구하기 
print(data_minmax['MEDV'].describe())

print(data_minmax['MEDV'] > 0.5)

print(data_minmax[data_minmax['MEDV'] > 0.5]['MEDV'])

print(data_minmax[data_minmax['MEDV'] > 0.5]['MEDV'].count())