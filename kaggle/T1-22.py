# T1-22. Time-Series4 (Weekly data)
# 주어진 데이터(basic2.csv)에서 주 단위 Sales의 합계를 구하고, 
# 가장 큰 값을 가진 주와 작은 값을 가진 주의 차이를 구하시오(절대값)

import pandas as pd
# data = pd.read_csv('bigData-main/basic2.csv', parse_dates=['Date'], index_col=0)
# print(data.head())

# 아래 코드를 한줄로 표현함
data = pd.read_csv('bigData-main/basic2.csv')
# Date 컬럼의 data 타입을 datetime으로 변경 
data['Date'] = pd.to_datetime(data['Date'])
print(data.head())

# Date 컬럼을 index로 
data = data.set_index('Date')
print(data.head())

# 주 단위 W
# 2주 단위 2W
# 월 단위 M
data_w = data.resample('W').sum()
print(data_w.head())

print(data_w['Sales'].max() - data_w['Sales'].min())