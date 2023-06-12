# 문제 : MEDV 칼럼을 기준으로 30번째로 큰 값을 1번 ~ 29번째로 큰 값에 적용한 뒤, 
# MEDV 컬럼의 평균값, 중위값, 최솟값, 최댓값 순으로 한줄에 출력하시오

import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

print(data['MEDV'].head(3))

# MEDV를 내림차순으로 정렬 후, data_new에 저장
data_new = data['MEDV'].sort_values(ascending = False)
print(data_new.head(30))

# 30번째로 큰값 출력
print(data_new.iloc[29])

# 1 ~ 29번째로 큰 값 확인
print(data_new.iloc[0:28])

# 1 ~ 29번째로 큰 값들은 41.7로 변경
data_new.iloc[0:28] = data_new.iloc[29]

# 1 ~ 29번째로 큰 값 확인
print(data_new.iloc[0:28])

# 컬럼의 평균값, 중위값, 최솟값, 최댓값 순으로 한줄에 출
print(data_new.mean(), data_new.median(), data_new.min(), data_new.max())