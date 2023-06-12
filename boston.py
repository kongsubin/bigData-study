import pandas as pd

data = pd.read_csv("./bigData-main/boston.csv")

print(data.head())

# data 변수를 MEDV칼럼 기준으로 오름차순 정렬
print("\ndata 변수를 MEDV칼럼 기준으로 오름차순 정렬")
print(data.sort_values(by = 'MEDV', ascending = True))

# MEDV 컬럼만 추출하기
print("\nMEDV 컬럼만 추출하기")
print(data.sort_values(by = 'MEDV', ascending = True)['MEDV'])

# MEDV 컬럼만 상위 10개 추출하기
print("\nMEDV 컬럼만 상위 10개 추출하기")
print(data.sort_values(by = 'MEDV', ascending = True)['MEDV'].head(10))