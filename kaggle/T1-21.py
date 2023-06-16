# T1-21. Binning Data / #비닝 #구간나누기
# 나이 구간 나누기
# basic1 데이터 중 'age'컬럼 이상치를 제거하고, 동일한 개수로 나이 순으로 3그룹으로 나눈 뒤 각 그룹의 중앙값을 더하시오
# (이상치는 음수(0포함), 소수점 값)

import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')

# age 컬럼 이상치를 제거 > 음수 제거 
print(data['age'].unique())
print(data.shape)
data_new = data[data['age'] > 0]
print(data_new.shape)

# age 컬럼 이상치를 제거 > 소숫점 제거 
print(round(data_new['age'], 0))
data_new = data_new[data_new['age'] == round(data_new['age'], 0)]
print(data_new.shape)

# 분할 기준 보기 
print(pd.qcut(data_new['age'], q=3))

# 구간 분할 
data_new['range'] = pd.qcut(data_new['age'], q=3, labels=['g1', 'g2', 'g3'])
print(data_new)

# 수량 비교 
print(data_new['range'].value_counts())

# 중앙값
m1 = data_new[data_new['range'] == 'g1']['age'].median()
m2 = data_new[data_new['range'] == 'g2']['age'].median()
m3 = data_new[data_new['range'] == 'g3']['age'].median()

print(m1+m2+m3)