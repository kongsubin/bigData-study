# 성인 인구조사 소득 예측
# age: 나이
# workclass: 고용 형태
# fnlwgt: 사람의 대표성을 나타내는 가중치(final weight)
# education: 교육 수준
# education.num: 교육 수준 수치
# marital.status: 결혼 상태
# occupation: 업종
# relationship: 가족 관계
# race: 인종
# sex: 성별
# capital.gain: 양도 소득
# capital.loss: 양도 손실
# hours.per.week: 주당 근무 시간
# native.country: 국적
# income: 수익 (예측해야 하는 값)

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
    
df = pd.read_csv("bigData-main/adult.csv")
x_train, x_test, y_train, y_test = exam_data_load(df, target='income', null_name='?')

x_train = x_train.reset_index().drop(columns='index')
y_train = y_train.reset_index().drop(columns='index')
x_test = x_test.reset_index().drop(columns='index')
y_test = y_test.reset_index().drop(columns='index')

####### 데이터 탐색
# 1. 데이터 행렬 확인 
print("데이터 행렬 확인")
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 2. 데이터 세부정보 및 내용 확인 
print("\n데이터 세부정보 및 내용 확인")
print(x_train.info())
print(x_train.head().T)
print("\n데이터 세부정보 및 내용 확인")
print(y_train.info())
print(y_train.head())

# 3. 기초통계량 확인하기 
print(x_train.describe())

# 4. object 컬럼 확인 
print(x_train['workclass'].unique())
print(x_train['occupation'].unique())
print(x_train['native.country'].unique())
print(x_train['education'].unique())
print(x_train['marital.status'].unique())
print(x_train['relationship'].unique())
print(x_train['race'].unique())
print(x_train['sex'].unique())


####### 데이터 전처리
# 1. 필요없는 컬럼 삭제 
x_test_id = x_test['id']
x_train = x_train.drop(columns='id')
x_test = x_test.drop(columns='id')
y_train = y_train.drop(columns='id')

# 2. 결측치 확인
# 결측치는 최빈값과 차이가 크면 최빈값으로, 값이 비슷하면 별도의 값으로 대체함
print(x_train.isnull().sum())
print(x_train['workclass'].value_counts())
print(x_train['occupation'].value_counts())
print(x_train['native.country'].value_counts())
x_train['workclass'] = x_train['workclass'].fillna(x_train['workclass'].mode()[0])
x_train['occupation'] = x_train['occupation'].fillna("nan")
x_train['native.country'] = x_train['native.country'].fillna(x_train['native.country'].mode()[0])
print(x_train.isnull().sum())

print(x_test.isnull().sum())
print(x_test['workclass'].value_counts())
print(x_test['occupation'].value_counts())
print(x_test['native.country'].value_counts())
x_test['workclass'] = x_test['workclass'].fillna(x_test['workclass'].mode()[0])
x_test['occupation'] = x_test['occupation'].fillna("nan")
x_test['native.country'] = x_test['native.country'].fillna(x_test['native.country'].mode()[0])
print(x_test.isnull().sum())

# 3. 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x_train['workclass'] = encoder.fit_transform(x_train['workclass']) 
x_train['occupation'] = encoder.fit_transform(x_train['occupation']) 
x_train['native.country'] = encoder.fit_transform(x_train['native.country']) 
x_train['education'] = encoder.fit_transform(x_train['education']) 
x_train['marital.status'] = encoder.fit_transform(x_train['marital.status']) 
x_train['relationship'] = encoder.fit_transform(x_train['relationship']) 
x_train['race'] = encoder.fit_transform(x_train['race']) 
x_train['sex'] = x_train['sex'].replace('Male', 0).replace('Female', 1)

x_test['workclass'] = encoder.fit_transform(x_test['workclass']) 
x_test['occupation'] = encoder.fit_transform(x_test['occupation']) 
x_test['native.country'] = encoder.fit_transform(x_test['native.country']) 
x_test['education'] = encoder.fit_transform(x_test['education']) 
x_test['marital.status'] = encoder.fit_transform(x_test['marital.status']) 
x_test['relationship'] = encoder.fit_transform(x_test['relationship']) 
x_test['race'] = encoder.fit_transform(x_test['race']) 
x_test['sex'] = x_test['sex'].replace('Male', 0).replace('Female', 1)

# 4. 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
print(x_train.head())
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
print(x_test.head())

# 5. y값 바꾸기
condition = y_train['income'] == ">50K"
y_train.loc[condition, 'income'] = 1
y_train.loc[~condition, 'income'] = 0
y_train['income'] = y_train['income'].astype('int64')

print(x_train.info())
print(y_train)
 
 
####### 학습 및 평가
from sklearn.model_selection import train_test_split
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(x_train, y_train, test_size=0.2)

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_TRAIN, Y_TRAIN)
# from sklearn.metrics import accuracy_score
# pred = model.predict(X_TEST)
# print('accuracy score:', (accuracy_score(Y_TEST, pred)))

# from xgboost import XGBClassifier
# model = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='error')
# model.fit(X_TRAIN, Y_TRAIN)
# from sklearn.metrics import accuracy_score
# pred = model.predict(X_TEST)
# print('accuracy score:', (accuracy_score(Y_TEST, pred)))

from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='error')
model.fit(x_train, y_train)
pred = model.predict(x_test)

pred = pd.DataFrame(pred)
print(pred)

print(pd.concat([x_test_id, pred], axis=1).rename(columns={0:'income'}))
result = pd.concat([x_test_id, pred], axis=1).rename(columns={0:'income'})
result.to_csv('data/t2-3.csv', index=False)