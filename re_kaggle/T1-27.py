# T1-27. 바닐라라떼 5점, 카페라떼 3점, 아메리카노 2점, 나머지 0점이다 총 메뉴의 점수를 더한 값은?

import pandas as pd 
data = pd.read_csv('bigData-main/payment.csv')

data['menu'] = data['menu'].str.replace(" ", "")

print(data['menu'].str.contains('바닐라라떼').sum()*5 + data['menu'].str.contains('카페라떼').sum()*3 + data['menu'].str.contains('아메리카노').sum()*2)