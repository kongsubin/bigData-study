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

data_new = data.groupby(['year', 'month'])['Sales'].sum()
data_new = pd.DataFrame(data_new)
data_new.columns = ['sum']
print(data_new.head())

data_new = data_new.sort_values(by=['year', 'sum'], ascending=False)