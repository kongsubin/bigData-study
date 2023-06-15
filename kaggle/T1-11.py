# min-max스케일링 기준 상하위 5% 구하기
# 주어진 데이터에서 'f5'컬럼을 min-max 스케일 변환한 후, 상위 5%와 하위 5% 값의 합을 구하시오

import pandas as pd
data = pd.read_csv('./bigData-main/basic1.csv', encoding='utf-8')
print(data.head(5))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['f5_1'] = scaler.fit_transform(data[['f5']])
print(data.head(5))

lower = data['f5_1'].quantile(0.05)
print(lower)
upper = data['f5_1'].quantile(0.95)
print(upper)

print(lower + upper)