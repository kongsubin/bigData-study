# 주어진 데이터에서 이상치(소수점 나이)를 찾고 올림, 내림, 버림(절사)했을때 
# 3가지 모두 이상치 'age' 평균을 구한 다음 모두 더하여 출력하시오
# 데이터셋 : basic1.csv

import pandas as pd
import numpy as np
data = pd.read_csv('bigData-main/basic1.csv')

data = data[(data['age'] - np.floor(data['age']) != 0 )]
data_ceil = np.ceil(data['age']).mean()
data_floor = np.floor(data['age']).mean()
data_trunc = np.trunc(data['age']).mean()

print(data_ceil + data_floor + data_trunc)