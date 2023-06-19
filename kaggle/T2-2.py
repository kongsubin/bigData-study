# 당뇨병 여부 판단
# 이상치 처리 (Glucose, BloodPressure, SkinThickness, Insulin, BMI가 0인 값)
# [참고]작업형2 문구
# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())
# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가
# 데이터 파일 읽기 예제
# import pandas as pd
# X_test = pd.read_csv("data/X_test.csv")
# X_train = pd.read_csv("data/X_train.csv")
# y_train = pd.read_csv("data/y_train.csv")
# 사용자 코딩
# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)


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
    

##### 데이터 탐색 
# 데이터 가져오기 
df = pd.read_csv("bigData-main/diabetes.csv")
x_train, x_test, y_train, y_test = exam_data_load(df, target='Outcome')

# 데이터 행열 확인 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 데이터 정보 확인 
print(x_train.info())
print(y_train.info())

# 데이터 내용 확인
print(x_train.head(1).T)
print(y_train.head(1).T)

# 기초 통계량 확인 
print(x_train.describe())


##### 데이터 전처리 
# 1. 필요없는 컬럼 삭제
x_test_id = x_test['id']
print(x_test_id.head())
x_train = x_train.drop(columns='id')
x_test = x_test.drop(columns='id')
y_train = y_train.drop(columns='id')

print(x_train.head())
print(x_test.head())
print(y_train.head())

# 2. 결측치 처리
print(x_train.isnull().sum())

# 3. 이상값 처리 
print("\ntrain")
print('Glucose', len(x_train[x_train['Glucose'] == 0]))
print('BloodPressure', len(x_train[x_train['BloodPressure'] == 0]))
print('SkinThickness', len(x_train[x_train['SkinThickness'] == 0]))
print('Insulin', len(x_train[x_train['Insulin'] == 0]))
print('BMI', len(x_train[x_train['BMI'] == 0]))
# 포도당 이상치 제거 
print(x_train[x_train['Glucose'] == 0].index)
glucose_index = x_train[x_train['Glucose'] == 0].index
print(x_train.shape, y_train.shape)
x_trian = x_train.drop(index=glucose_index, axis=0, inplace=True)
y_trian = y_train.drop(index=glucose_index, axis=0, inplace=True)
print(x_train.shape, y_train.shape)
# 나머지 이상치 제거 
x_train.loc[x_train['BloodPressure'] == 0,'BloodPressure'] = x_train['BloodPressure'].mean()
x_train.loc[x_train['SkinThickness'] == 0,'SkinThickness'] = x_train['SkinThickness'].mean()
x_train.loc[x_train['Insulin'] == 0,'Insulin'] = x_train['Insulin'].mean()
x_train.loc[x_train['BMI'] == 0,'BMI'] = x_train['BMI'].mean()
print("\ntrain 이상치 제거 후")
print('Glucose', len(x_train[x_train['Glucose'] == 0]))
print('BloodPressure', len(x_train[x_train['BloodPressure'] == 0]))
print('SkinThickness', len(x_train[x_train['SkinThickness'] == 0]))
print('Insulin', len(x_train[x_train['Insulin'] == 0]))
print('BMI', len(x_train[x_train['BMI'] == 0]))

print("\ntest")
print('Glucose', len(x_test[x_test['Glucose'] == 0]))
print('BloodPressure', len(x_test[x_test['BloodPressure'] == 0]))
print('SkinThickness', len(x_test[x_test['SkinThickness'] == 0]))
print('Insulin', len(x_test[x_test['Insulin'] == 0]))
print('BMI', len(x_test[x_test['BMI'] == 0]))
# 나머지 이상치 제거 
x_test.loc[x_test['BloodPressure'] == 0,'BloodPressure'] = x_test['BloodPressure'].mean()
x_test.loc[x_test['SkinThickness'] == 0,'SkinThickness'] = x_test['SkinThickness'].mean()
x_test.loc[x_test['Insulin'] == 0,'Insulin'] = x_test['Insulin'].mean()
x_test.loc[x_test['BMI'] == 0,'BMI'] = x_test['BMI'].mean()
print("\ntest 이상치 제거 후")
print('Glucose', len(x_test[x_test['Glucose'] == 0]))
print('BloodPressure', len(x_test[x_test['BloodPressure'] == 0]))
print('SkinThickness', len(x_test[x_test['SkinThickness'] == 0]))
print('Insulin', len(x_test[x_test['Insulin'] == 0]))
print('BMI', len(x_test[x_test['BMI'] == 0]))

# 4. 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x_train[cols] = scaler.fit_transform(x_train[cols])
x_test[cols] = scaler.fit_transform(x_test[cols])

##### 학습 및 평가
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)

from sklearn.model_selection import train_test_split 
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(x_train, y_train, test_size=0.2)
MODEL = XGBClassifier(n_estimators=100, max_depth=5)
MODEL.fit(X_TRAIN, Y_TRAIN)
Y_TEST_PREDICT = MODEL.predict(X_TEST)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_TEST_PREDICT, Y_TEST))

y_test_predict = model.predict(x_test)
result = pd.DataFrame(y_test_predict)
print(result.head())

x_test_id.reset_index(drop=True, inplace=True)
print(x_test_id)
print(result)


output = pd.concat([x_test_id, result], axis=1).rename(columns={0:'Outcome'})
print(output)

output.to_csv('data/t2-2.csv', index=False)