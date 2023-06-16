# Q. 수학, 영어 점수 중 사람과 과목에 상관없이 
# 90점 이상인 점수의 평균을 정수로 구하시오 (소수점 버림)

import pandas as pd
data = pd.DataFrame({'Name': {0: '김딴짓', 1: '박분기', 2: '이퇴근'},
                   '수학': {0: 90, 1: 93, 2: 85},
                   '영어': {0: 92, 1: 84, 2: 86},
                   '국어': {0: 91, 1: 94, 2: 83},})

print(data)

# 풀이
data = pd.melt(data, id_vars=['Name'], value_vars=['수학', '영어'])
print(data)

cond = data['value'] >= 90
print(int(data[cond]['value'].mean()))