# 데이터에서 IQR을 활용해 Fare컬럼의 이상치를 찾고, 이상치 데이터의 여성 수를 구하시오
# 강의 영상 : https://youtu.be/ipBW5D_UJEo
# 데이터셋 : titanic

import pandas as pd
data = pd.read_csv('bigData-main/titanic.csv')

q1 = data['Fare'].describe()['25%']
q3 = data['Fare'].describe()['75%']
iqr = q3 - q1
iqr_1 = q1 - iqr*1.5
iqr_3 = q3 + iqr*1.5

print(iqr_1, iqr_3)
data_new = data[(data['Fare'] > iqr_3) | (data['Fare'] < iqr_1)]
print(len(data_new[data_new['Sex'] == 'female']))