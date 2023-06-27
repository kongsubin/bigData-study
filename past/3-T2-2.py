# 대학원 입학 예측(회귀) / 리더보드 제출 아님
# 예측할 값(target): "Chance of Admit "
# 평가: r2
# data(3개): t2-2-X_train, t2-2-y_train, t2-2-X_test
# 제출 형식(Serial No.-> id, 예측 값 -> target)
# id,target
# 28,0.741696
# 76,0.779616
# 151,0.897247

import pandas as pd
x_train = pd.read_csv('past/data/t2-2-X_train.csv')
y_train = pd.read_csv('past/data/t2-2-y_train.csv')
x_test = pd.read_csv('past/data/t2-2-X_test.csv')

print(x_train.shape, y_train.shape, x_test.shape)

print(x_train.head(3))
print(y_train.head(3))
print(x_test.head(3))

print(x_train.info())
print(y_train.info())
print(x_test.info())

# 필요없는 컬럼 삭제
id = x_test['Serial No.']
x_train = x_train.drop(columns=['Serial No.'])
x_test = x_test.drop(columns=['Serial No.'])
print(x_train.head(3))
print(x_test.head(3))

# 결측치 확인
print(x_train.isnull().sum())
print(x_test.isnull().sum())

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)

# 학습
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

xtr,xva,ytr,yva = train_test_split(x_train, y_train['Chance of Admit '], test_size=0.2)
model = RandomForestRegressor()
model.fit(xtr, ytr)
pred = model.predict(xva)
print(r2_score(yva, pred))

model = XGBRegressor(n_estimators=100, max_depth=5)
model.fit(xtr, ytr)
pred = model.predict(xva)
print(r2_score(yva, pred))

model_ = RandomForestRegressor()
model_.fit(x_train, y_train['Chance of Admit '])
pred = model_.predict(x_test)
pred = pd.DataFrame(pred).rename(columns={0:'target'})

print(pd.concat([id, pred], axis=1).rename(columns={'Serial No.':'id'}))
result = pd.concat([id, pred], axis=1).rename(columns={'Serial No.':'id'})

result.to_csv('past/data/t2-2-submission.csv', index=False)
