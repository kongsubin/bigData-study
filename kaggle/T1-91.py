# t1-91-sigmoid
# [심화 학습] basic1 데이터에서 f2 값에 시그모이드 함수(그림 내 수식)를 적용하고, 
# f4가 ISFJ인 (변경된)f2의 합을 구하시오! (소수점 둘째자리까지 출력, 반올림)
import pandas as pd
import numpy as np

data = pd.read_csv('bigData-main/basic1.csv')
print(data.info())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data['f2_sigmoid'] = data['f2'].apply(sigmoid)

cond = data['f4'] == 'ISFJ'
print(round(data[cond]['f2_sigmoid'].sum(),2))