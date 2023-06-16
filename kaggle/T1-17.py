# T1-17. Time-Series1 / #시계열데이터 #datetime 
# 2022년 5월 sales의 중앙값을 구하시오
# 데이터셋 : basic2.csv

import pandas as pd
data = pd.read_csv('bigData-main/basic2.csv')
print(data.head())

print(data.info())
data['Date'] = pd.to_datetime(data['Date'])
print(data.info())

data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
print(data.head())

print(type(data))

print(data[(data['year'] == 2022) & (data['month'] == 5)]['Sales'].median())