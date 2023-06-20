# 주어진 데이터에서 결측치가 80%이상 되는 컬럼은(변수는) 삭제하고, 
# 80% 미만인 결측치가 있는 컬럼은 'city'별 중앙값으로 값을 대체하고 'f1'컬럼의 평균값을 출력하세요!
# 데이터셋 : basic1.csv 

import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')
print(data.head())

print(data.isnull().sum())
data = data.drop(columns='f3')
print(data.isnull().sum())