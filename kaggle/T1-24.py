# T1-24. Time-Series5 (Lagged Feature) 시차 데이터 생성
# 주어진 데이터(basic2.csv)에서 "pv"컬럼으로 1일 시차(lag)가 있는 새로운 컬럼을 만들고
# (예: 1월 2일에는 1월 1일 pv데이터를 넣고, 1월 3일에는 1월 2일 pv데이터를 넣음),
# 새로운 컬럼의 1월 1일은 다음날(1월 2일)데이터로 결측치를 채운 다음, 
# Events가 1이면서 Sales가 1000000이하인 조건에 맞는 새로운 컬럼 합을 구하시오
# 데이터셋 : basic2.csv

import pandas as pd
data = pd.read_csv('bigData-main/basic2.csv')

print(data.isnull().sum())

#1일 차이가 나는 시차 특성 만들기
data['previous_PV'] = data['PV'].shift(1)
print(data.head())

# 1일 씩 미뤘음으로 가장 앞이 결측값이 됨 (바로 뒤의 값으로 채움)
data['previous_PV'] = data['previous_PV'].fillna(method = 'bfill')
print(data.head())

# 조건에 맞는 1일 이전 PV의 합
cond = (data['Events'] == 1) & (data['Sales'] <= 1000000)
print(data[cond]['previous_PV'].sum())