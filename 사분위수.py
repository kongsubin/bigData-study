# 문제 : CHAS 칼람과 RAD 칼럼을 제외한 칼럼에 한해서 컬럼별 IQR 값 구하기 

import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

print(data.shape)
# CHAS 칼럼과 RAD 칼럼 삭제 
data_col12 = data.drop(columns = ['CHAS', 'RAD'])
print(data_col12.shape)

# 기초통계량 정보를 저장
data_col12_desc = data_col12.describe()
print(data_col12_desc)

# data_col12_desc 변수에서 4번행, 6번행 데이터 가져오기 
print(data_col12_desc.iloc[[4, 6]])

# 세로축으로 변경
print(data_col12_desc.iloc[[4, 6]].T)

data_col12_desc_T = data_col12_desc.iloc[[4, 6]].T
print(data_col12_desc_T)

# IQR = 75% - 25%
print(data_col12_desc_T['75%'] - data_col12_desc_T['25%'])