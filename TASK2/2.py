# 문제 : 고객 891명에 대한 학습용 데이터를 이용하여 생존여부를 예측하는 모형만들기 
# 이를 평가용 데이터에 저굥ㅇ하여 승객의 생존 여부 예측값을 다음과 같은 형식의 csv 파일로 생성하셈 
# 모델의 성능은 ROC-AUC 평가지표에 땨라 매겨짐
# 제출 형식
# PassengerId, Servivied
# 892, 0
# 893, 1

##############################################
#                 데이터 탐색                   #
##############################################
# 1. 데이터 가져오기
import pandas as pd
x_train = pd.read_csv('bigData-main/titanic_x_train.csv', encoding='CP949')
y_train = pd.read_csv('bigData-main/titanic_y_train.csv')
x_test = pd.read_csv('bigData-main/titanic_x_test.csv', encoding='CP949')

# x_train, x_test의 상위 1개 행 확인
print(x_train.head(1).T)
print(x_test.head(1).T)

# y_train 확인
print(y_train.head(5))

# 2. 각 데이터의 행열 갯수 확인
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

# 3. 요약정보 확인하기 
print(x_train.info())

# object로 되어있는 칼람들 확인 
# 성별 컬럼의 중복 제거한 값과 개수 확인
print(x_train['성별'].unique().size)
print(x_train['성별'].unique())
# 선착장 칼럼의 중복 제거한 값과 개수 확인
print(x_train['선착장'].unique().size)
print(x_train['선착장'].unique())
# 나머지 3개 컬럼 
print(x_train['승객이름'].unique().size)
print(x_train['티켓등급'].unique().size)
print(x_train['객실번호'].unique().size)

# 4. 기초 통계량 확인하기
print(x_train.describe().T)

# 5. 독립변수와 종속변수의 관계 확인하기 
# x_train, y_train을 세로 방향으로 통합한후, data변수에 저장
data = pd.concat([x_train, y_train], axis=1)
# 성별 칼럼에 따라 survivied의 평균값 구하기 
print(data.groupby(['성별'])['Survived'].mean())
# 티켓등급 칼럼에 따라 survivied의 평균값 구하기 
print(data.groupby(['티켓등급'])['Survived'].mean())
# 선착장 칼럼에 따라 survivied의 평균값 구하기 
print(data.groupby(['선착장'])['Survived'].mean())


##############################################
#                데이터 전처리                  #
##############################################
# 1. 필요한 컬럼 삭제하기 
x_test_passenger_id = x_test['PassengerId']

x_train = x_train.drop(columns=['PassengerId'])
y_train = y_train.drop(columns=['PassengerId'])
x_test = x_test.drop(columns=['PassengerId'])

print(x_train.head(1).T)
print(y_train.head())

# 티켓번호 칼럼 > 681건의 중복 제거된 값을 가짐 
print(x_train['티켓번호'].unique().size)
x_train = x_train.drop(columns=['티켓번호'])
x_test = x_test.drop(columns=['티켓번호'])
# 승객이름 칼럼 > 삭제 
x_train = x_train.drop(columns=['승객이름'])
x_test = x_test.drop(columns=['승객이름'])

# 2. 결측치 처리하기 > 나이 객실번호 선착장
print(x_train.isnull().sum())

# 나이가 생존여부와 상관성 확인 
print(data[['나이', 'Survived']].corr())
# -0.077로 상관관계 매우 낮음 > 삭제 
x_train = x_train.drop(columns='나이')
x_test = x_test.drop(columns='나이')

# 객실번호 결측치 : 687건
# 중복제외, 148건 존재 > 삭제 
print(x_train['객실번호'].unique().size)
x_train = x_train.drop(columns='객실번호')
x_test = x_test.drop(columns='객실번호')

# 선착장 > 결측치 2개 > 최빈값으로 처리
print(x_train.groupby(['선착장'])['선착장'].count())
x_train['선착장'] = x_train['선착장'].fillna('S')
print(x_train['선착장'].isnull().sum())

# 3. 범주형 변수를 인코딩 
# 성별 컬럼을 연속형 변수로 변환 (수동 인코딩)
x_train['성별'] = x_train['성별'].replace('male', 0).replace('female', 1)
x_test['성별'] = x_test['성별'].replace('male', 0).replace('female', 1)

# 선착장 칼람에 원핫 인코딩 수행
선착장_dummy = pd.get_dummies(x_train['선착장'], drop_first=True, dtype=int).rename(columns={'Q':'선착장Q', 'S':'선착장S'})
# 기존 x_train의 우측에 선착장_dummy변수를 덧붙여, x_train에 다시 저장
x_train = pd.concat([x_train, 선착장_dummy], axis=1)
print(x_train.head())
x_train = x_train.drop(columns='선착장')

# 선착장 칼람에 원핫 인코딩 수행
선착장_dummy = pd.get_dummies(x_test['선착장'], drop_first=True, dtype=int).rename(columns={'Q':'선착장Q', 'S':'선착장S'})
# 기존 x_train의 우측에 선착장_dummy변수를 덧붙여, x_test 다시 저장
x_test = pd.concat([x_test, 선착장_dummy], axis=1)
print(x_test.head())
x_test = x_test.drop(columns='선착장')

# 4. 파생변수 만들기 
# 현제자매배우자수 + 부모자식수 칼람 합치기 
x_train['가족수'] = x_train['형제자매배우자수'] + x_train['부모자식수']
print(x_train[['형제자매배우자수', '부모자식수', '가족수']].head(10))
x_train = x_train.drop(columns=['형제자매배우자수', '부모자식수'])

x_test['가족수'] = x_test['형제자매배우자수'] + x_test['부모자식수']
print(x_test[['형제자매배우자수', '부모자식수', '가족수']].head(10))
x_test = x_test.drop(columns=['형제자매배우자수', '부모자식수'])


##############################################
#                학습 및 평가                   #
##############################################
# 1. 데이터 분리하기 
from sklearn.model_selection import train_test_split

X_TRAIN, X_TEST, Y_TRIAN, Y_TEST = train_test_split(x_train, y_train, test_size=0.2)
# 분리된 데이터의 행렬 구조 확인
print(X_TRAIN.shape)
print(X_TEST.shape)
print(Y_TRIAN.shape)
print(Y_TEST.shape)

# 2. 데이터 학습 및 하이퍼 파라미터 튜닝 
# XGBClassifer : 일반적으로 성능이 잘나옴 
from xgboost import XGBClassifier 
# 기본
model1 = XGBClassifier(eval_metric = 'error')
model1.fit(X_TRAIN, Y_TRIAN)

# 하이퍼파라미터 적용
model2 = XGBClassifier(n_estimators = 100, max_depth=5, eval_metric='error')
model2.fit(X_TRAIN, Y_TRIAN)



# 3. 결과 예측
y_test_predict = pd.DataFrame(model2.predict(x_test)).rename(columns={0:'Survived'})
print(pd.DataFrame(y_test_predict).head())

# 승객이 사망할 확률 : pd.DataFrame(model.predict_proba(x_test))[0]
# 승객이 생존할 확률 : pd.DataFrame(model.predict_proba(x_test))[1]

# 학습이 완료된 모델을 통해 Y_TEST 예측. 평가지표 계산용
Y_TEST_PREDICT1 = pd.DataFrame(model1.predict(X_TEST))
Y_TEST_PREDICT2 = pd.DataFrame(model2.predict(X_TEST))

# 4. 모델 평가 
# 평가 비교
from sklearn.metrics import roc_auc_score
# model1
print(roc_auc_score(Y_TEST, Y_TEST_PREDICT1))
# model2
print(roc_auc_score(Y_TEST, Y_TEST_PREDICT2))


##############################################
#                 결과 제출                    #
##############################################
print(pd.concat([x_test_passenger_id, y_test_predict], axis=1))

final = pd.concat([x_test_passenger_id, y_test_predict], axis=1)

final.to_csv('data/21800491_2.csv', index=False)

