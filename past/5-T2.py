# 빅데이터 분석기사 5회 실기 기출 유형
# [가격 예측] 중고 자동차
# 자동차 가격을 예측해주세요!
# 예측할 값(y): price
# 평가: RMSE (Root Mean Squared Error)
# data: train.csv, test.csv
# [컴피티션 제출 양식] 리더보드 제출용
# 제출 형식: submission.csv파일을 아래와 같은 형식(수치형)으로 제출
# (id는 test의 index임)
# id,price
# 0,11000
# 1,20500
# 2,19610
# ...    
# 1616,11995


import pandas as pd 
train = pd.read_csv('past/data/train_5.csv')
x_test = pd.read_csv('past/data/test_5.csv')
x_train = train.drop(columns=['price'])

print(x_train.head())
print(x_test.head())

print(x_train.info())

# 결측치 확인
print(x_train.isnull().sum())
print(x_test.isnull().sum())

# 인코딩
print(x_train['model'].unique())
print(x_train['transmission'].unique())
print(x_train['fuelType'].unique())
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

x_train['model'] = encoder.fit_transform(x_train['model'])
x_train['transmission'] = encoder.fit_transform(x_train['transmission'])
x_train['fuelType'] = encoder.fit_transform(x_train['fuelType'])

x_test['model'] = encoder.fit_transform(x_test['model'])
x_test['transmission'] = encoder.fit_transform(x_test['transmission'])
x_test['fuelType'] = encoder.fit_transform(x_test['fuelType'])

# 스케일링



