# 문제 : 결측치를 삭제한 표준편차 값과 결측치를 평균값으로 대치한 후에 산출된 표준편차 차이를 양의 정수로 출력 
import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

# data의 결측치 여부 확인
print(data.isnull())

# 결측치 갯수 세기
print(data.isnull().sum())

# ************ 평균값 ************ #
# 결측치가 있는 RM 컬럼만 추출 
data_mean = data['RM'].copy()
print(data_mean.isnull().sum())

# 계산한 평균값을 rm_mean에 저장
rm_mean = data_mean.mean()
print(rm_mean)

# RM컬럼만 추출된 data_mean 변수에서 평균값으로 결측치를 대치 
# inplace 옵션 : F -> 변경된 값을 data_mean변수에 반영x 
print(data_mean.fillna(rm_mean, inplace = False)) # 단순히 확인하는 용도이므로 False
print(data_mean.isnull().sum())

# inplace 옵션 : T -> 변경된 값을 data_mean변수에 반영o
data_mean.fillna(rm_mean, inplace = True)
print(data_mean.isnull().sum())

# ************ 삭제 ************ #
# 결측치가 있는 RM 컬럼만 추출 
data_del = data['RM'].copy()

# data_del에서도 결측치 개수 확인
print(data_del.isnull().sum())

# data_del 변수의 행/열 구조
print(data_del.shape)

# data_del 변수의 저장과 함께 결측치 삭제 
data_del.dropna(inplace = True)

# data_del 변수의 행/열 구조
print(data_del.shape)

# 506 -> 491 로 줄어듬 : 삭제되었다는 증거 
print(data_del.isnull().sum())

# ************ 표준편차 ************ #
print(data_mean.std())
print(data_del.std())

# ********** 표준편차차이 ********** #
print(abs(data_mean.std()-data_del.std()))