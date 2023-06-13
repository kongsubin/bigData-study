# 문제 : boston의 전체 컬럼에 대해서 중복을 제거한 유니크한 값을 구한후, 
# 칼람별 유니크한 값의 개수를 기준으로 평균값을 구하기 


import pandas as pd
data = pd.read_csv("./bigData-main/boston.csv")


### 첫번째 
# 컬럼의 목록 확인
print(data.columns)
data_col = data.columns
print(data_col.size)

# CHAS 칼람에서 유일한 값들의 목록 확인
print(data['CHAS'].unique())

# 데이터 프레임 타입으로 변환
print(pd.DataFrame(data['CHAS'].unique()))

# 결과 건수 계산
print(pd.DataFrame(data['CHAS'].unique()).count())

# 유니크한 개수를 정수형으로 변환하여 출력 
print(int(pd.DataFrame(data['CHAS'].unique()).count()[0]))

# 전체 컬럼에 대한 유니크한 개수 더하여 평균 내기
print(
    (int(pd.DataFrame(data['CRIM'].unique()).count()[0]) + 
     int(pd.DataFrame(data['ZN'].unique()).count()[0]) +
     int(pd.DataFrame(data['INDUS'].unique()).count()[0]) +
     int(pd.DataFrame(data['CHAS'].unique()).count()[0]) +
     int(pd.DataFrame(data['NOX'].unique()).count()[0]) +
     int(pd.DataFrame(data['RM'].unique()).count()[0]) +
     int(pd.DataFrame(data['AGE'].unique()).count()[0]) +
     int(pd.DataFrame(data['DIS'].unique()).count()[0]) +
     int(pd.DataFrame(data['RAD'].unique()).count()[0]) +
     int(pd.DataFrame(data['TAX'].unique()).count()[0]) +
     int(pd.DataFrame(data['PTRATIO'].unique()).count()[0]) +
     int(pd.DataFrame(data['B'].unique()).count()[0]) +
     int(pd.DataFrame(data['LSTAT'].unique()).count()[0]) +
     int(pd.DataFrame(data['MEDV'].unique()).count()[0])
     ) / data_col.size
)

### 두번째 
sum = 0
for i in data_col :
    sum = sum + int(pd.DataFrame(data[i].unique()).count()[0])
    
print(sum / data_col.size)