# T1-24. Time-Series5 (Lagged Feature) 시차 데이터 생성
# 주어진 데이터(basic2.csv)에서 "pv"컬럼으로 1일 시차(lag)가 있는 새로운 컬럼을 만들고
# (예: 1월 2일에는 1월 1일 pv데이터를 넣고, 1월 3일에는 1월 2일 pv데이터를 넣음),
# 새로운 컬럼의 1월 1일은 다음날(1월 2일)데이터로 결측치를 채운 다음, 
# Events가 1이면서 Sales가 1000000이하인 조건에 맞는 새로운 컬럼 합을 구하시오
# 데이터셋 : basic2.csv

import pandas as pd
data = pd.read_csv('bigData-main/basic2.csv')
data['pv_pre'] = data['PV'].shift(1)
data['pv_pre'] = data['pv_pre'].fillna(method='bfill')
condition = (data['Events'] == 1) & (data['Sales'] <= 1000000)

print(data[condition]['pv_pre'].sum())
