# T1-14. Multi Index & Groupby / #멀티인덱스 #정렬 #인덱스리셋 #상위값
# city와 f4를 기준으로 f5의 평균값을 구한 다음, f5를 기준으로 상위 7개 값을 모두 더해 출력하시오 (소수점 둘째자리까지 출력)
# - 데이터셋 : basic1.csv 
import pandas as pd
data = pd.read_csv('bigData-main/basic1.csv')

data_new = pd.DataFrame(data.groupby(by=['city', 'f4'])['f5'].mean())
data_new = data_new.sort_values(by='f5', ascending=False)

print(round(data_new['f5'].head(7).sum(), 2))
