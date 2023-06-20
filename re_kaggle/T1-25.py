# basic1 데이터에서 f4가 E로 시작하면서 부산에 살고 20대인 사람은 몇 명일까요?

import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')

cond1 = data['f4'].str.startswith('E')
cond2 = data['city'] == '부산'
cond3 = (data['age'] < 30) & (data['age'] > 19)

print(len(data[cond1 & cond2 & cond3]))