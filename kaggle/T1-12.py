# T1-12. top10-bottom10 / #그룹핑 #정렬 #상하위값
# 주어진 데이터에서 상위 10개 국가의 접종률 평균과 하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요 
# (단, 100%가 넘는 접종률 제거, 소수 첫째자리까지 출력)

import pandas as pd
data = pd.read_csv('bigData-main/covid-vaccination-vs-death_ratio.csv')
print(data.head())
print(data['country'].unique())

data_new = data.groupby(['country'])['ratio'].max()
print(data_new.head())

data_new = pd.DataFrame(data_new)

data_new.sort_values(by='ratio', ascending=False, inplace=True)
print(data_new.head())
print(data_new[data_new['ratio'] < 100].head(10))

print(data_new[data_new['ratio'] < 100].head(10).mean().iloc[0])
print(data_new[data_new['ratio'] < 100].tail(10).mean().iloc[0])
head = data_new[data_new['ratio'] < 100].head(10).mean().iloc[0]
tail = data_new[data_new['ratio'] < 100].tail(10).mean().iloc[0]

print(round((head-tail), 1))