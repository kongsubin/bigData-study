# T1-30. 12월 25일 결제 금액(price)은 12월 총 결제금액의 몇 %인가? (정수로 출력)
import pandas as pd
data = pd.read_csv('bigData-main/payment.csv')
data['date'] = pd.to_datetime(data['date'], format="%Y%m%d")

data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

cond_12 = data['month'] == 12
cond_25 = data['day'] == 25
price_12 = data[cond_12]['price'].sum()
price_12_25 = data[cond_12 & cond_25]['price'].sum()

print(int(price_12_25 / price_12 * 100))