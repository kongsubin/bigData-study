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

# object 컬럼 확인 
print(x_train['workclass'].unique())
print(x_train['education'].unique())
print(x_train['marital.status'].unique())
print(x_train['occupation'].unique())
print(x_train['relationship'].unique())
print(x_train['race'].unique())
print(x_train['sex'].unique())
print(x_train['native.country'].unique())


####### 데이터 전처리
# 1. 필요없는 컬럼 삭제 
x_test_id = x_test['id']
x_train = x_train.drop(columns='id')
x_test = x_test.drop(columns='id')
y_train = y_train.drop(columns='id')

# 2. 결측치 확인
print(x_train.isnull().sum()
