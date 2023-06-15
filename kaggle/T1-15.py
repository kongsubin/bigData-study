# 주어진 데이터 셋에서 age컬럼 상위 20개의 데이터를 구한 다음 
# f1의 결측치를 중앙값으로 채운다.
# 그리고 f4가 ISFJ와 f5가 20 이상인 
# f1의 평균값을 출력하시오!

# - 데이터셋 : basic1.csv 

import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')
print(data.head())

data = data.sort_values(by='age', ascending=False)
print(data.head())
print(data.shape)

data_new = data.head(20)
print(data_new.shape)

print(data_new.isnull().sum())
data_new['f1'] = data_new['f1'].fillna(data_new['f1'].median())
print(data_new.isnull().sum())

print(data_new[(data_new['f4'] == 'ISFJ') & (data_new['f5'] >= 20)]['f1'].mean())


# 중앙값인지 평균인지~~ 좀 잘봐라~~