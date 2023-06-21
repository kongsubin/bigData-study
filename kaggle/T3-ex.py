# 주어진 데이터는 고혈압 환자 치료 전후의 혈압이다. 
# 해당 치료가 효과가 있는지 대응(쌍체)표본 t-검정을 진행하시오

# t-검정 : 모집단의 분산이나 표준편차를 알지 못할 때, 
# 표본으로부터 추정된 분산이나 표준편차를 이용하여 두 모집단의 평균의 차이를 알아보는 검정 방법

# 귀무가설(H0):  
# μ >= 0
# 대립가설(H1):  
# μ < 0
# 
# μ = (치료 후 혈압 - 치료 전 혈압)의 평균
# 유의수준: 0.05
# 
# μ의 표본평균은?(소수 둘째자리까지 반올림)
# 검정통계량 값은?(소수 넷째자리까지 반올림)
# p-값은?(소수 넷째자리까지 반올림)
# 가설검정의 결과는? (유의수준 5%)

import pandas as pd
from scipy import stats
df = pd.read_csv('bigData-main/high_blood_pressure.csv')
print(df.head())

df['diff'] = df['bp_post'] - df['bp_pre']

#1
print(round(df['diff'].mean(),2))

#2 
# st - 검정통계량, pv - p value(작아야 채택))
st, pv = stats.ttest_rel(df['bp_post'], df['bp_pre'], alternative="less")
print(round(st,4))

#3
print(round(pv,4))

#4 귀무가설 기각, 대립가설 채택 (0.0016 < 0.05)


# -6.12
# -3.0002
# 0.0016