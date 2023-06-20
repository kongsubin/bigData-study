# T1-21. Binning Data / #비닝 #구간나누기
# 나이 구간 나누기
# basic1 데이터 중 'age'컬럼 이상치를 제거하고, 
# 동일한 개수로 나이 순으로 3그룹으로 나눈 뒤 각 그룹의 중앙값을 더하시오
# (이상치는 음수(0포함), 소수점 값)

import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')

condition = (data['age'] > 0) & (data['age'] == round(data['age'], 0))
data = data[condition]

data['range'] = pd.qcut(data['age'], q=3, labels={'g1', 'g2', 'g3'})

g1 = data[data['range'] == 'g1']['age'].median()
g2 = data[data['range'] == 'g2']['age'].median()
g3 = data[data['range'] == 'g3']['age'].median()

print(g1+g2+g3)