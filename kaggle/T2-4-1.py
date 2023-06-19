# 집 값 예측
# 예측할 변수 ['SalePrice']
# 평가: rmse, r2

# rmse는 낮을 수록 좋은 성능
# r2는 높을 수록 좋은 성능

# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv("bigData-main/train.csv")
x_train, x_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')
x_train = x_train.reset_index().drop(columns='index')
y_train = y_train.reset_index().drop(columns='index')
x_test = x_test.reset_index().drop(columns='index')
y_test = y_test.reset_index().drop(columns='index')

######### EDA
# 1. 데이터 행열 확인하기 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 2. 데이터 상세정보 확인 
print(x_train.info())
print(x_train.head())
print(x_train.value_counts())

# y값을 보면, 값을 예측하는 것을 알 수 있음 > Regression 
print(y_train.info())
print(y_train.head())
y_target = y_train['SalePrice']


######### 전처리
# 1. 필요없는 컬럼 삭제 
x_train = x_train.select_dtypes(exclude=['object'])
print(x_train.head())
x_test = x_test.select_dtypes(exclude=['object'])


# 2. 결측치 정보 확인
print(x_train.isnull().sum())
print(x_test.isnull().sum())

print(x_train['LotFrontage'].describe())
print(x_train['MasVnrArea'].mode()[0])
print(x_train['GarageYrBlt'].describe())

x_train['LotFrontage'] = x_train['LotFrontage'].fillna(x_train['LotFrontage'].mean())
x_train['MasVnrArea'] = x_train['MasVnrArea'].fillna(x_train['MasVnrArea'].mode()[0])
x_train['GarageYrBlt'] = x_train['GarageYrBlt'].fillna(x_train['GarageYrBlt'].mean())

x_test['LotFrontage'] = x_test['LotFrontage'].fillna(x_test['LotFrontage'].mean())
x_test['MasVnrArea'] = x_test['MasVnrArea'].fillna(x_test['MasVnrArea'].mode()[0])
x_test['GarageYrBlt'] = x_test['GarageYrBlt'].fillna(x_test['GarageYrBlt'].mean())

print(x_train.isnull().sum())
print(x_test.isnull().sum())

# 3. 라벨 인코딩 > 생략

# 4. 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)


######### 학습 및 평가
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

X_tr, X_val, Y_tr, Y_val = train_test_split(x_train, y_target, test_size=0.2)
model_xgb = XGBRegressor()
model_xgb.fit(X_tr, Y_tr)
pred = model_xgb.predict(X_val)
print(r2_score(pred, Y_val))
print(mean_squared_error(pred, Y_val))

model_rf = RandomForestRegressor()
model_rf.fit(X_tr, Y_tr)
pred = model_rf.predict(X_val)
print(r2_score(pred, Y_val))
print(mean_squared_error(pred, Y_val))

model = XGBRegressor()
model.fit(x_train, y_target)
predict = model.predict(x_test)

predict = pd.DataFrame(predict)
print(predict)

print(pd.concat([y_test['Id'], predict], axis=1).rename(columns={0:'SalePrice'}))
output = pd.concat([y_test['Id'], predict], axis=1).rename(columns={0:'SalePrice'})
output.to_csv('data/t2-4-1.csv', index=False)

print("RMSE : " + str(mean_squared_error(y_test['SalePrice'], predict)))
print("R2 : " + str(r2_score(y_test['SalePrice'], predict)))