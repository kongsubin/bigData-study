# T1-30. 12월 25일 결제 금액(price)은 12월 총 결제금액의 몇 %인가? (정수로 출력)
import pandas as pd
data = pd.read_csv('bigData-main/payment.csv')
print(data.info())

data['date'] = pd.to_datetime(data['date'], format="%Y%m%d")

data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
print(data.head())

sum_12 = data[data['month'] == 12]['price'].sum()
print(sum_12)

sum_12_25 = data[(data['month'] == 12) & (data['day'] == 25)]['price'].sum()
print(sum_12_25)

print(int(sum_12_25/sum_12 * 100))

# 다른 풀이
cond1 = data['date'].dt.month == 12
cond2 = data['date'].dt.day == 25
result = sum(data[cond1 & cond2]['price']) / sum(data[cond1]['price'])
print(int(result*100))