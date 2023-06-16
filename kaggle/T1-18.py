# T1-18. Time-Series2 / #주말 #평일 #비교 #시계열데이터
# 주어진 데이터에서 2022년 5월 주말과 평일의 sales컬럼 평균값 차이를 구하시오 (소수점 둘째자리까지 출력, 반올림)
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
data['dayofweek'] = data['Date'].dt.dayofweek
print(data.head())

week = data[(data['year'] == 2022) & (data['month'] == 5) & (data['dayofweek'] > 4)]['Sales'].mean()
print(week)

day = data[(data['year'] == 2022) & (data['month'] == 5) & (data['dayofweek'] < 5)]['Sales'].mean()
print(day)

print(round(abs(week - day), 2))