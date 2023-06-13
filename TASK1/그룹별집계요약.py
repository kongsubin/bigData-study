# 문제 : TAX 컬럼이 TAX컬럼의 중위값보다 큰 데이터를 대상으로, CHAS 컬럼과 RAD 컬럼순으로 
# 그룹을 지은 후, 각 그룹의 데이터 개수 출력. 칼럼별 데이터 개수는 COUNT로 출력


import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

# TAX컬럼의 중위값
tax_median = data['TAX'].median()
print(tax_median)

# TAX컬럼의 중위값보다 큰 데이터를 대상
print(data['TAX'] > tax_median)

# CHAS 컬럼과 RAD 컬럼
print(data[data['TAX'] > tax_median][['CHAS', 'RAD']])
data_new = data[data['TAX'] > tax_median][['CHAS', 'RAD']]

# CHAS 컬럼 종류 확인
print(data_new['CHAS'].unique())

# CHAS 컬럼 종류 확인
print(data_new['RAD'].unique())

# 그룹화 : groupby(그룹화할 컬럼들)[수행할대상].수행할작업()
data_new2 = data_new.groupby(['CHAS', 'RAD'])['RAD'].count()
print(data_new2)

# 비어있는 칼럼 이름에 COUNT를 채우기 
# 우선 데이터 타입부터 변경
print(type(data_new2)) # 시리즈 > 데이터프레임 타입으로 변경
data_new3 = pd.DataFrame(data_new2)

# 카운트 연산을 수행한 결과 컬럼 이름을 COUNT로
data_new3.columns = ['COUNT']
print(data_new3)