# 문제 : DIS 칼람을 표준화 철도로 변환한 후, 0.4보다 크면서 0.6보다 작은 값들에 대한 평균을 구하기
# 단, 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지 출력 

import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 표준변환 실행
data_stdd = scaler.fit_transform(data)

# 데이터 타입확인
print(type(data_stdd))
# 데이터프레임으로 변환
data_stdd = pd.DataFrame(data_stdd, columns=data.columns)

print((data_stdd['DIS'] > 0.4) & (data_stdd['DIS'] < 0.6))
print(data_stdd[(data_stdd['DIS'] > 0.4) & (data_stdd['DIS'] < 0.6)]['DIS'])

data_stdd = data_stdd[(data_stdd['DIS'] > 0.4) & (data_stdd['DIS'] < 0.6)]

# 평균값 구하기
print(data_stdd['DIS'].mean())

# 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지 출력 
print(round(data_stdd['DIS'].mean(), 2))