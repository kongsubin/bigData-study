# T1-11. min-max scaling / #스케일링 #상하위값
# min-max스케일링 기준 상하위 5% 구하기
# 주어진 데이터에서 'f5'컬럼을 min-max 스케일 변환한 후, 상위 5%와 하위 5% 값의 합을 구하시오

import pandas as pd
data = pd.read_csv('./bigData-main/basic1.csv', encoding='utf-8')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['f5'] = scaler.fit_transform(data[['f5']])

a = data['f5'].quantile(0.05)
b = data['f5'].quantile(0.95)

print(a+b)