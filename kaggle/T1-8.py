# 주어진 데이터 셋에서 'f2' 컬럼이 1인 조건에 해당하는 데이터의 'f1'컬럼 누적합을 계산한다. 
# 이때 발생하는 누적합 결측치는 바로 뒤의 값을 채우고, 
# 누적합의 평균값을 출력한다. (단, 결측치 바로 뒤의 값이 없으면 다음에 나오는 값을 채워넣는다)
# 데이터셋 : basic1.csv

import pandas as pd 
data = pd.read_csv('bigData-main/basic1.csv')

data = data[data['f2'] == 1]
data['cumsum'] = data['f1'].cumsum()

data['cumsum'] = data['cumsum'].fillna(method='bfill')
print(data['cumsum'].mean())