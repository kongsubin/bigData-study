# [분류] 여행 보험 패키지 상품을 구매할 확률 값을 구하시오 / 리더보드 제출용
# 예측할 값(y): TravelInsurance (여행보험 패지지를 구매 했는지 여부 0:구매안함, 1:구매)
# 평가: roc-auc 평가지표
# data: t2-1-train.csv, t2-1-test.csv

# 제출 형식
# id,TravelInsurance
# 0,0.3
# 1,0.48
# 2,0.3
# 3,0.83

import pandas as pd
train = pd.read_csv('past/data/t2-1-train.csv')
print(train.head())

x_test = pd.read_csv('past/data/t2-1-test.csv')

# x, y 나누기 
y_train = train['TravelInsurance']
y_train.columns = ['TravelInsurance']
x_train = train.drop(columns='TravelInsurance')

# EDA
# 행렬 확인
print(x_train.shape, y_train.shape, x_test.shape)

# 데이터 정보 확인
print(x_train.head(2).T)
print(x_train.info())

# 전처리
# 필요없는 칼람 삭제
x_test_id = x_test['id']
print(x_test_id)
x_test = x_test.drop(columns='id')
x_train = x_train.drop(columns='id')

print(x_train.shape, y_train.shape, x_test.shape)

# 결측치 확인 
print(x_train.isnull().sum())
print(x_test.isnull().sum())

x_train['AnnualIncome'] = x_train['AnnualIncome'].fillna(x_train['AnnualIncome'].mean())
x_test['AnnualIncome'] = x_test['AnnualIncome'].fillna(x_test['AnnualIncome'].mean())

print(x_train.isnull().sum())
print(x_test.isnull().sum())

# 인코딩 
print(x_train['Employment Type'].unique())
print(x_train['GraduateOrNot'].unique())
print(x_train['FrequentFlyer'].unique())
print(x_train['EverTravelledAbroad'].unique())

x_train['Employment Type'] = x_train['Employment Type'].replace('Private Sector/Self Employed', 0).replace('Government Sector', 1).replace('Casual employment', 3)
x_train['GraduateOrNot'] = x_train['GraduateOrNot'].replace('No', 0).replace('Yes', 1)
x_train['FrequentFlyer'] = x_train['FrequentFlyer'].replace('No', 0).replace('Yes', 1)
x_train['EverTravelledAbroad'] = x_train['EverTravelledAbroad'].replace('No', 0).replace('Yes', 1)

print(x_test['Employment Type'].unique())
print(x_test['GraduateOrNot'].unique())
print(x_test['FrequentFlyer'].unique())
print(x_test['EverTravelledAbroad'].unique())

x_test['Employment Type'] = x_test['Employment Type'].replace('Private Sector/Self Employed', 0).replace('Government Sector', 1).replace('Casual employment', 3)
x_test['GraduateOrNot'] = x_test['GraduateOrNot'].replace('No', 0).replace('Yes', 1)
x_test['FrequentFlyer'] = x_test['FrequentFlyer'].replace('No', 0).replace('Yes', 1)
x_test['EverTravelledAbroad'] = x_test['EverTravelledAbroad'].replace('No', 0).replace('Yes', 1)

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(x_train.info())

x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)


# 학습
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
print(x_train)
print(train['TravelInsurance'])
xtr, xval, ytr, yval = train_test_split(x_train, train['TravelInsurance'], test_size=0.2)

model = RandomForestClassifier()
model.fit(xtr, ytr)
pred = model.predict(xval)
print(roc_auc_score(yval, pred))

model = XGBClassifier()
model.fit(xtr, ytr)
pred = model.predict(xval)
print(roc_auc_score(yval, pred))

model_ = XGBClassifier()
model_.fit(x_train, train['TravelInsurance'])
pred = model_.predict(x_test)
proba = model_.predict_proba(x_test)
proba = pd.DataFrame(proba)
print(proba)

result = pd.concat([x_test_id, proba[1]], axis=1).rename(columns={1:'TravelInsurance'})
print(result)

result.to_csv('past/data/t2-1-submission.csv', index=False)