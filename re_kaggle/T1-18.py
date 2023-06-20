# T1-18. Time-Series2 / #주말 #평일 #비교 #시계열데이터
# 주어진 데이터에서 2022년 5월 주말과 평일의 sales컬럼 평균값 차이를 구하시오 (소수점 둘째자리까지 출력, 반올림)
# 데이터셋 : basic2.csv

import pandas as pd
data = pd.read_csv('bigData-main/basic2.csv')
data['Date'] = pd.to_datetime(data['Date'])

data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data['dayofweek'] = data['Date'].dt.dayofweek

condition = (data['year'] == 2022) & (data['month'] == 5)
cond_week = data['dayofweek'] > 4
cond_day = data['dayofweek'] < 5

week_mean = data[condition & cond_week]['Sales'].mean()
day_mean = data[condition & cond_day]['Sales'].mean()

print(round(abs(week_mean - day_mean), 2))