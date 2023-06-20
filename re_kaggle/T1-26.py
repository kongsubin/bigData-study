# T1-26. menu컬럼에 "라떼" 키워드가 있는 데이터의 수는?

import pandas as pd
data = pd.read_csv('bigData-main/payment.csv')

print(data['menu'].str.contains('라떼').sum())