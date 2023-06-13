# 문제 : TAX 컬럼을 오름차순으로 정렬한 결과, 내림차순으로 정렬한 결과 구하기 
# 각 순번에 맞는 오름차순값과 내림차순 값의 차이를 구해 분산 값 출력 
import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")

data_asc = data['TAX'].copy()
data_desc = data['TAX'].copy()

# 오름차순정렬
print(data_asc.sort_values(ascending = True))
data_asc.sort_values(inplace=True)
# 내림차순정렬
print(data_desc.sort_values(ascending = False)) 
data_desc.sort_values(ascending = False, inplace=True)

# 행번호 똑같게 재설정
data_asc.reset_index(drop=True, inplace=True) #drop=True : 현재 인덱스 정보를 완전히 삭제 
print(data_asc)
data_desc.reset_index(drop=True, inplace=True)
print(data_desc)

# 칼럼 기준으로 통합하기
data_concat = pd.concat([data_asc, data_desc], axis = 1) 
# 만약 data_asc 아래로 data_desc 덧붙이는 경우 : axis = 0
print(data_concat)

# 최종 결과 얻기
# 첫번째 컬럼 추출
print(data_concat.iloc[:,0])
# 두번째 컬럼 추출
print(data_concat.iloc[:,1])

# 차이값 구하기 
data_concat['diff'] = abs(data_concat.iloc[:,0] - data_concat.iloc[:,1])
print(data_concat)

# 차이값의 분산 출력 
diff_var = data_concat['diff'].var()
print(diff_var)