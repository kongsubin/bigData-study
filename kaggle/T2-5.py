# Insurance_Starter (Tutorial)
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
    
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)
    
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv("bigData-main/insurance.csv")
x_train, x_test, y_train, y_test = exam_data_load(df, target='charges')
x_train = x_train.reset_index().drop(columns='index')
y_train = y_train.reset_index().drop(columns='index')
x_test = x_test.reset_index().drop(columns='index')
y_test = y_test.reset_index().drop(columns='index')


########## EDA
# 1. 행열 확인
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 2. 세부정보 확인
print(x_train.info())
print(x_train.head())

# y 값 예측 > Regression 
print(y_train.info())
print(y_train.head())

# 3. object 확인 
print(x_train['sex'].unique())
print(x_train['smoker'].unique())
print(x_train['region'].unique())

# 4. 기초통계량 확인
print(x_train.describe())

########## 전처리
# 1. 필요없는 행 삭제 
x_test_id = x_test['id']
x_train = x_train.drop(columns='id')
x_test = x_test.drop(columns='id')
y_train = y_train.drop(columns='id')

print(x_train.head())
print(x_test.head())
print(y_train.head())

# 2. 결측치 확인
print(x_train.isnull().sum())
print(x_test.isnull().sum())

# 3. 이상치 확인
print(x_train['age'].value_counts())
print(x_train[x_train['bmi'] <= 0])
print(x_train['children'].value_counts())

print(x_test['age'].value_counts())
print(x_test[x_test['bmi'] <= 0])
print(x_test['children'].value_counts())

# 4. 라벨 인코딩 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x_train['region'] = encoder.fit_transform(x_train['region'])
x_train['sex'] = x_train['sex'].replace('male', 0).replace('female', 1)
x_train['smoker'] = x_train['smoker'].replace('no', 0).replace('yes', 1)
print(x_train.head())
x_test['region'] = encoder.fit_transform(x_test['region'])
x_test['sex'] = x_test['sex'].replace('male', 0).replace('female', 1)
x_test['smoker'] = x_test['smoker'].replace('no', 0).replace('yes', 1)
print(x_test.head())

# 5. 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)


########## 학습 및 평가
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def rmse2(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print(y_train)

Xtr, Xval, Ytr, Yval = train_test_split(x_train, y_train, test_size=0.2)
model_xgb = XGBRegressor()
model_xgb.fit(Xtr, Ytr)
pred = model_xgb.predict(Xval)
print(r2_score(Yval, pred))
print(rmse2(Yval, pred))

model_rf = RandomForestRegressor()
model_rf.fit(Xtr, Ytr.values.ravel())
pred = model_rf.predict(Xval)
print(r2_score(Yval, pred))
print(rmse2(Yval, pred))

model = RandomForestRegressor()
model.fit(x_train, y_train.values.ravel())
prediction = model.predict(x_test)
prediction = pd.DataFrame(prediction)

print(pd.concat([x_test_id, prediction], axis=1).rename(columns={0:'charges'}))
output = pd.concat([x_test_id, prediction], axis=1).rename(columns={0:'charges'})
output.to_csv('data/t2-5.csv', index=False)
print(r2_score(y_test['charges'], prediction))
print(rmse2(y_test['charges'], prediction))