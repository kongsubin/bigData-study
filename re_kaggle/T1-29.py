# T1-29. 12월인 데이터 수는?
import pandas as pd
data = pd.read_csv('bigData-main/payment.csv')
data['date'] = pd.to_datetime(data['date'], format="%Y%m%d")
data['month'] = data['date'].dt.month
print(data[data['month'] == 12]['month'].count())