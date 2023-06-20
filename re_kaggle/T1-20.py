# T1-20. Combining Data / 데이터 #병합 #결합 / 고객과 궁합이 맞는 타입 매칭
# basic1 데이터와 basic3 데이터를 'f4'값을 기준으로 병합하고,
# 병합한 데이터에서 r2결측치를 제거한다음, 앞에서 부터 20개 데이터를 선택하고 'f2'컬럼 합을 구하시오

import pandas as pd
data1 = pd.read_csv('bigData-main/basic1.csv')
data3 = pd.read_csv('bigData-main/basic3.csv')

data = pd.merge(left=data1, right=data3, how="left", on='f4')

data = data.dropna(subset=['r2'])

print(data.head(20)['f2'].sum())