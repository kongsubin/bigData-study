# 문제 : ZN값의 평균값에서 표준편차의 1.5배보다 크거나 작은 ZN값의 합계를 구하기 
# 1. ZN 칼럼의 평균값과 표준편차 구하기
# 2. 평균값에서 1.5*표준편차보다 큰 값 구하기
# 3. 평균값에서 1.5*표준편차보다 작은 값 구하기 

import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

# ZN 컬럼의 평균값 
zn_mean = data['ZN'].mean()
# ZN 컬럼의 표준편차
zn_std = data['ZN'].std()

# 평균값에서 1.5*표준편차보다 큰 값 구하기
zn_max = zn_mean + (1.5 * zn_std)
print(zn_max)

# 평균값에서 1.5*표준편차보다 작은 값 구하기 
zn_min = zn_mean - (1.5 * zn_std)
print(zn_min)

# zn_max 보다 큰지 여부 확인
print(data['ZN'] > zn_max)

# 실제 숫자 확인
print(data[data['ZN'] > zn_max]) # 전체 컬럼들 출력됨
print(data[data['ZN'] > zn_max]['ZN']) # ZN 출력됨

zn_max2 = data[data['ZN'] > zn_max]['ZN']
print(zn_max2)

# 작은값 확인 
print(data[data['ZN'] < zn_min]) # Index: [] 출력됨 -> 없음

# zn_max2 변숫값의 합계 구하기
print(sum(zn_max2))