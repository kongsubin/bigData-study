# T1-19. Time-Series3 (monthly total) / #월별 #총계 #비교 #데이터값변경
# 주어진 데이터에서 2022년 월별 Sales 합계 중 가장 큰 금액과
# 2023년 월별 Sales 합계 중 가장 큰 금액의 차이를 절대값으로 구하시오.
# 단 Events컬럼이 '1'인경우 80%의 Salse값만 반영함
# (최종값은 소수점 반올림 후 정수 출력)

import pandas as pd
data = pd.read_csv('bigData-main/basic2.csv')
print(data.head())

print(data.info())
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

data_2022 = data[data['year'] == 2022]
data_2023 = data[data['year'] == 2023]

data_2022.loc[data_2022['Events'] == 1, 'Sales'] = data_2022['Sales']*0.8
data_2023.loc[data_2023['Events'] == 1, 'Sales'] = data_2023['Sales']*0.8

max_2022 = data_2022.groupby(['year', 'month'])['Sales'].sum().max()
max_2023 = data_2023.groupby(['year', 'month'])['Sales'].sum().max()

print(max_2022)
print(max_2023)

result = int(round(abs(max_2022 - max_2023), 0))
print(result)

