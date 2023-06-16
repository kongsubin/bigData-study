# T1-29. 12월인 데이터 수는?
import pandas as pd
data = pd.read_csv('bigData-main/payment.csv')
print(data.info())

data['date'] = pd.to_datetime(data['date'], format="%Y%m%d")
print(data.info())


print(sum(data['date'].dt.month == 12))