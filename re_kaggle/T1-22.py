# T1-22. Time-Series4 (Weekly data)
# 주어진 데이터(basic2.csv)에서 주 단위 Sales의 합계를 구하고, 
# 가장 큰 값을 가진 주와 작은 값을 가진 주의 차이를 구하시오(절대값)

import pandas as pd
data = pd.read_csv('bigData-main/basic2.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

sum = data.resample('W').sum()
print(abs(sum['Sales'].max() - sum['Sales'].min()))

