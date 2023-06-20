# t1-31 
# 수학, 영어, 국어 점수 중 사람과 과목에 상관없이 가장 상위 점수 5개를 모두 더하고 출력하시오.

import pandas as pd
data = pd.DataFrame({'Name': {0: '김딴짓', 1: '박분기', 2: '이퇴근'},
                   '수학': {0: 90, 1: 93, 2: 85},
                   '영어': {0: 92, 1: 84, 2: 86},
                   '국어': {0: 91, 1: 94, 2: 83},})



data_new = pd.melt(data, id_vars=['Name'])

result = data_new.sort_values(by='value', ascending=False).head(5)['value'].sum()
print(result)