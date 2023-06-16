# T1-13. Correlation / #상관관계
# 상관관계 구하기
# 주어진 데이터에서 상관관계를 구하고, quality와의 상관관계가 가장 큰 값과, 가장 작은 값을 구한 다음 더하시오!
# 단, quality와 quality 상관관계 제외, 소수점 둘째 자리까지 출력

import pandas as pd
data = pd.read_csv('bigData-main/winequality-red.csv')
print(data.head())

data_corr = data.corr()['quality']
print(data_corr[:-1])

max = abs(data_corr[:-1]).max()
min = abs(data_corr[:-1]).min()

print(max, min)
print(round((max + min), 2))