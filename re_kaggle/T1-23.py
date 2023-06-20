# T1-23. 중복 데이터 제거 Drop Duplicates
# f1의 결측치를 채운 후 age 컬럼의 중복 제거 전과 후의 'f1' 중앙값 차이를 구하시오
# 결측치는 f1의 데이터 중 내림차순 정렬 후 10번째 값으로 채움
# 중복 데이터 발생시 뒤에 나오는 데이터를 삭제함
# 데이터셋 : basic1.csv

import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')

na = data.sort_values(by='f1', ascending=False).head(10)['f1']
data['f1'] = data['f1'].fillna(na.iloc[9])

before = data['f1'].median()
data = data.drop_duplicates(subset='age')
after = data['f1'].median()

print(abs(before-after))