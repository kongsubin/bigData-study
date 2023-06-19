# T1-28. 시간(hour)이 13시 이전(13시 포함하지 않음) 
# 데이터 중 가장 많은 결제가 이루어진 날짜(date)는? (date 컬럼과 동일한 양식으로 출력)
import pandas as pd
data = pd.read_csv('bigData-main/payment.csv')
print(data.head())

data_new = data[data['hour'] < 13]
print(data_new.head())

print(data_new['date'].value_counts())
print(data_new['date'].value_counts().index[0])