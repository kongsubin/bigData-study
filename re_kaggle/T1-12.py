# T1-12. top10-bottom10 / #그룹핑 #정렬 #상하위값
# 주어진 데이터에서 상위 10개 국가의 접종률 평균과 하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요 
# (단, 100%가 넘는 접종률 제거, 소수 첫째자리까지 출력)

import pandas as pd
data = pd.read_csv('bigData-main/covid-vaccination-vs-death_ratio.csv')
print(data.head())

data_new = pd.DataFrame(data.groupby(by='country')['ratio'].max())
print(data_new.head())

data_new = data_new.sort_values(by='ratio', ascending=False)
print(data_new.head())

condition = data_new['ratio'] < 100 
a = data_new.loc[condition, 'ratio'].head(10).mean()
b = data_new.loc[condition, 'ratio'].tail(10).mean()
print(a)
print(b)
print(round(abs(a-b), 1))