# T1-16. Variance / #분산 #결측치전후값차이
# 주어진 데이터 셋에서 f2가 0값인 데이터를 age를 기준으로 오름차순 정렬하고
# 앞에서 부터 20개의 데이터를 추출한 후 
# f1 결측치(최소값)를 채우기 전과 후의 분산 차이를 계산하시오 (소수점 둘째 자리까지)

# - 데이터셋 : basic1.csv 


import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')
print(data.head())

data_new = data[data['f2']==0].sort_values('age', ascending=True)
print(data_new.head())

print(data_new.shape)
data_new = data_new[:20]
print(data_new.shape)

before = data_new['f1'].var()
print(before)

data_new['f1'] = data_new['f1'].fillna(data_new['f1'].min())
after = data_new['f1'].var()
print(after)

print(round(abs(before-after), 2))