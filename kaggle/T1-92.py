# t1-92-relu
# basic1 데이터에서 age 값에 ReLU 함수를 적용하고, 
# city가 부산인 (변경된)age의 평균을 구하시오! (소수점 둘째자리까지 출력, 절사)\import pandas as pd

import pandas as pd
import numpy as np

data = pd.read_csv('bigData-main/basic1.csv')
print(data.info())

# ReLU 함수
# h(x) =

# x (x > 0) : 0보다 크면 x
# 0 (x <= 0) : 0이거나 0보다 작으면 0


def relu(x):
    return np.maximum(0, x)

data['age'] = data['age'].apply(relu)

cond = data['city'] == "부산"
print(int(data[cond]['age'].mean()))