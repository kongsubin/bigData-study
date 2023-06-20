# T1-17. Time-Series1 / #시계열데이터 #datetime 
# 2022년 5월 sales의 중앙값을 구하시오
# 데이터셋 : basic2.csv

import pandas as pd
data = pd.read_csv('bigData-main/basic2.csv')
data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month

median = data[(data['year'] == 2022) & (data['month'] == 5)]['Sales'].median()
print(median)