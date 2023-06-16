# T1-23. 중복 데이터 제거 Drop Duplicates
# f1의 결측치를 채운 후 age 컬럼의 중복 제거 전과 후의 'f1' 중앙값 차이를 구하시오
# 결측치는 f1의 데이터 중 내림차순 정렬 후 10번째 값으로 채움
# 중복 데이터 발생시 뒤에 나오는 데이터를 삭제함
# 데이터셋 : basic1.csv

import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')
print(data.head())

top10 = data['f1'].sort_values(ascending=False).iloc[9]
print(top10)
data['f1'] = data['f1'].fillna(top10)

# 중복 제거 전 중앙 값
before = data['f1'].median()
print(before)

# 중복 제거
print(data.shape)
data = data.drop_duplicates(subset=['age'])
print(data.shape)

# 중복 제거 후 중앙 값
after = data['f1'].median()
print(after)

# 차이 출력
print(abs(before - after))