import pandas as pd

# read csv file 
data = pd.read_csv('./bigData-main/mtcars.csv')

# data 변수의 행/열 확인
print("data 변수의 행/열 확인")
print(data.shape)

# data 변수의 열 이름 확인
print("data 변수의 열 이름 확인")
print(data.columns)

# 기초통계량 구하기
print("기초통계량 구하기")
print(data.describe())
print("hp 수치형 변수의 기초통계량 구하기")
print(data['hp'].describe())

#데이터 중복 없애기
print("gear컬럼에서 데이터 중복 없애기")
print(data['gear'].unique())

# 상관관계 구하기 
# print(data.corr())

##############################################
#             종속변수과 독립변수 구분             #
##############################################
X = data.drop(columns = 'mpg')
Y = data['mpg']
print("\n독립변수 X의 상위 3개 행 확인")
print(X.head(3))
print("\n독립변수 Y의 상위 3개 행 확인")
print(Y.head(3))

##############################################
#                데이터 전처리                  #
##############################################
######## 불필요한 열 삭제 (자동차의 모델명의 의미를 지닌 Unnamed: 0 열 삭제 )
X = X.iloc[:,1:]
print("\n\n불필요한 열 삭제 확인")
print(X.head(3))


######## 1. 결측값 처리하기 (문제에서 하라고 한 것 아니면 빅분기 시험에서 결측치를 삭제하는 방법을 선택해서는 안됨!!!!!!)
## 결측치 여부 확인 (True가 1이므로 sum을 해서 갯수 확인 )
print("\n\n결측치 여부 확인")
print(X.isnull().sum())
## 평균값으로 대치 
X_cyl_mean = X['cyl'].mean()
print("\ncyl 평균값으로 대치")
print(X_cyl_mean)
X['cyl'] = X['cyl'].fillna(X_cyl_mean)
print("\n다시 결측치 여부 확인")
print(X.isnull().sum())
## 중위값 대치 
X_qsec_mean = X['qsec'].median()
print("\nqsec 평균값으로 대치")
print(X_qsec_mean)
X['qsec'] = X['qsec'].fillna(X_qsec_mean)
print("\n다시 qsec 결측치 여부 확인")
print(X['qsec'].isnull().sum())
## 결측치를 임의의 값으로 교체 
## X['qsec'] = X['qsec'].fillna(100)
## 결측치 삭제 
## X.drop(columns = ['열이름'])


######## 2. 잘못된 값을 올바르게 바꾸기
## gear 열의 값을 확인
print("\n\ngear 열의 값을 확인")
print(X['gear'].unique())
## *3 -> 3, *5 -> 5 로 변경
print("\n*3 -> 3, *5 -> 5 로 변경")
print(X["gear"].replace('*3', '3').replace('*5', '5'))
## 저장 
X["gear"] = X["gear"].replace('*3', '3').replace('*5', '5')
print("\n잘못된 값을 올바르게 바꾼 gear 열의 값을 확인")
print(X['gear'].unique())


######## 3. 이상값 처리
### 3-1. 사분범위 활용 
X_describe = X.describe()
print(X_describe)

# 75%행 및 25%행만 출력
print("\n75%행 및 25%행만 출력")
print(X_describe.loc['75%'], "\n", X_describe.loc['25%'])
X_iqr = X_describe.loc['75%'] - X_describe.loc['25%']
print("\nX_iqr")
print(X_iqr)

print("\n@@사분위수를 활용한 최대 이상값 변경@@")
# 75%값 + 1.5*IQR
print("\n75%값 + 1.5*IQR")
print(X_describe.loc['75%'] + (1.5 * X_iqr))
# 각 열의 최댓값 확인 후, 최대 경계값을 넘는지 확인 -> 있는 경우에는 이상값임 
print("\n각 열의 최댓값 확인 후, 최대 경계값을 넘는지 확인 ")
print(X_describe.loc['max'])

# cyl 열 값이 14를 초과하는 값 추출하기 
print("\ncyl 열 값이 14를 초과하는 값 추출하기 ")
print(X.loc[X['cyl'] > 14])
# 인덱스가 14이고 열이 cyl인 값을 14로 변경
X.loc[14, 'cyl'] = 14
# X 변수에서 인덱스 14, 열이 cyl인 값 확인
print("\n인덱스가 14이고 열이 cyl인 값을 14로 변경")
print(X.loc[14, 'cyl'])

# hp 열 값이 305.25를 초과하는 값 추출하기 
print("\nhp 열 값이 305.25를 초과하는 값 추출하기 ")
print(X.loc[X['hp'] > 305.25])
# 인덱스가 30이고 열이 hp 값을 305.25로 변경
X.loc[30, 'hp'] = 305.25
# X 변수에서 인덱스 30, 열이 hp 값 확인
print("\n인덱스가 30이고 열이 hp인 값을 305.25로 변경")
print(X.loc[30, 'hp'])

print("\n@@사분위수를 활용한 최소 이상값 변경@@")
print("\n25%값 - 1.5*IQR")
print(X_describe.loc['25%'] - (1.5 * X_iqr))
# 각 열의 최솟값 확인 후, 최소 경계값을 넘는지 확인 -> 있는 경우에는 이상값임 
print("\n각 열의 최솟값 확인 후, 최소 경계값을 넘는지 확인 ")
print(X_describe.loc['min'])

### 3-2. 평균 + 1.5*표준편차, 평균 - 1.5*표준편차 
print("\n@@평균과 표준편차를 활용한 최대 이상값 변경@@")

## 데이터와 열을 전달하면 이상값 정보가 출력되는 outlier() 함수 제작
def outlier(data, column):
    mean = data[column].mean()
    std = data[column].std()
    lowest = mean - (std * 1.5)
    highest = mean + (std * 1.5)
    print('최소 경계값 : ', lowest, ' 최대 경계값 : ', highest)
    outlier_index = data[column][(data[column] < lowest) | (data[column] > highest)].index
    return outlier_index

print("\nqsec의 이상값 확인")
print(outlier(X, 'qsec'))
print("인덱스 24, 열이 qsec인 값 확인 : ")
print(X.loc[24, 'qsec'])
# 인덱스 24, 열이 qsec인 값을 42.396로 변경
print("인덱스 24, 열이 qsec인 값을 42.396로 변경")
X.loc[24, 'qsec'] = 42.396
# 인덱스 24, 열이 qsec인 값 확인
print(X.loc[24, 'qsec'])

print("\ncarb의 이상값 확인")
print(outlier(X, 'carb'))
print("인덱스 29, 30, 열이 carb인 값 확인 : ")
print(X.loc[[29, 30], 'carb'])
# 인덱스 29, 30, 열이 carb인 값을 5.235로 변경
print("인덱스 29, 30, 열이 carb인 값을 5.235로 변경")
X.loc[[29, 30], 'carb'] = 5.235
# 인덱스 [29, 30], 열이 carb인 값 확인
print(X.loc[[29, 30], 'carb'])

######## 4. 데이터 스케일링 
## 4-1. 표준크기변환 평균0, 표준편차1
from sklearn.preprocessing import StandardScaler
# X변수에서 qsec 열만 추출한 후, temp 변수에 저장하기.
temp = X[['qsec']]
# StandardScaler 함수 호출하여 표준 크기변환 기능을 갖는 scaler라는 객체 만들기. 
scaler = StandardScaler()
print("표준크기변환 수행한 qsec 결과 출력")
# 표준 크기변환하는 scaler에게 fit_transform 명령으로 temp 변수의 크기변환 요청하기. 
print(scaler.fit_transform(temp))
# 표준 크기변환을 수행한 결과를 변수에 저장
qsec_s_scaler = pd.DataFrame(scaler.fit_transform(temp))
# qsec_s_scaler 변수의 기초 통계량 확인
print("\nqsec_s_scaler 변수의 기초 통계량 확인")
print(qsec_s_scaler.describe())
# 크기변환한 변수를 사용하고 싶으면 아래처럼 저장해서 사용가능
# X['qsec'] = pd.DataFrame(scaler.fit_transform(temp))

## 4-2. 최소최대크기변환 최솟값0, 최댓값1 분포로 변환 
from sklearn.preprocessing import MinMaxScaler
# X변수에서 qsec 열만 추출한 후, temp 변수에 저장하기.
temp = X[['qsec']]
scaler = MinMaxScaler()
# 최소최대크기변환을 수행한 결과를 변수에 저장
qsec_m_scaler = pd.DataFrame(scaler.fit_transform(temp))
print("최소최대크기변환 수행한 qsec 결과 출력")
print(qsec_m_scaler)
# qsec_m_scaler 변수의 기초 통계량 확인
print("\nqsec_m_scaler 변수의 기초 통계량 확인")
print(qsec_m_scaler.describe())

## 4-3. 로버스트크기변환 중앙값0, 사분범위1 (이상값의 영향을 받지x, 일반적으로 활용하는 변환기법)
from sklearn.preprocessing import RobustScaler
# X변수에서 qsec 열만 추출한 후, temp 변수에 저장하기.
temp = X[['qsec']]
scaler = RobustScaler()
# 로버스트크기변환을 수행한 결과를 변수에 저장
qsec_r_scaler = pd.DataFrame(scaler.fit_transform(temp))
print("로버스트크기변환을 수행한 qsec 결과 출력")
print(qsec_r_scaler)
# qsec_r_scaler 변수의 기초 통계량 확인
print("\qsec_r_scaler 변수의 기초 통계량 확인")
print(qsec_r_scaler.describe()) # 사분위수 값 : 75% - 25%

######## 5. 데이터타입 변경 
# X변수의 요약정보 확인
print("\nX변수의 요약정보 확인")
print(X.info())
# 범주형이 연속형 데이터 타입으로 되어있거나, 그 반대의 경우 astype()함수를 통해 재설정
# 전진기어 개수를 의히마는 gear 열이 object 타입으로 설정됨 
# gear 열의 데이터 타입을 int64로 변경한 후, 다시 gear 열에 저장
print("\ngear 열의 데이터 타입을 int64로 변경한 후, 다시 gear 열에 저장")
X['gear'] = X['gear'].astype('int64')
# gear 열의 데이터 타입 확인하기
print("gear 열의 데이터 타입 확인하기")
print(X['gear'].dtype )

######## 6. 범주형을 수치형으로 변경 (인코딩)
## 6-1. 원핫 인코딩 
# 원래 의미 유지, 왜곡x
# am 컬럼의 auto -> 1, manual -> 0
print(X.head(5))
print("\nam 열에서 중복을 제거한 값 확인")
print(X['am'].unique())
print(pd.get_dummies(X['am'], dtype=int))
print("\n컬럼을 하나로!")
print(pd.get_dummies(X['am'], drop_first = True, dtype=int))

print("\nX변수의 데이터 타입 확인")
print(X.info())
# get_dummies는 기본적으로 범주형 데이터만 골라서 인코딩 처리 
print("\nX변수의 전체 열을 대상으로 원핫 인코딩 수행")
print(pd.get_dummies(X, drop_first = True, dtype=int))

## 6-2. 라벨 인코딩 
# 범주형 변수를 일련번호를 부여하는 방식 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
print(encoder.fit_transform(X['am']))
fruit = ['apple', 'banana', 'grape']
fruit_new = encoder.fit_transform(fruit)
print(fruit, fruit_new)

## 6-3. 수동 인코딩 
X['am_new'] = X['am'].replace('manual', 0).replace('auto',1)
print(X.head())

X = X.drop(columns='am')

######## 7. 파생변수 만들기
## 7-1. wt_class
# wt 열이 3.3보다 작은지 여부를 확인
print("\nwt 열이 3.3보다 작은지 여부를 확인")
print(X['wt'] < 3.3)
# 3.3기준으로 wt_classess 만들기 
condition = X['wt'] < 3.3
# X변수가 condition 조건 만족시, 0으로 저장 
X.loc[condition, 'wt_class'] = 0
# X변수가 condition 조건 만족하지 않으면, 1으로 저장 
X.loc[~condition, 'wt_class'] = 1
print("\nwt 열과 wt_class열 확인")
print(X[['wt', 'wt_class']])
# X 변수에서 wt 열 삭제 
X = X.drop(columns = 'wt')
print(X.head())

## 7-2. qsec 1mile 단위 변화하기 위해 생성 
X['qsec_4'] = X['qsec'] * 4
print("새로 만든 qsec_4확인")
print(X[['qsec', 'qsec_4']])
# X 변수에서 qsec 열 삭제 
X = X.drop(columns = 'qsec')
print(X.head())


##############################################
#                 데이터 분리                   #
##############################################
## 학습 데이터와 테스트 데이터 분리 
# 데이터 분리를 위해 train_test_split 함수 가져오기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

##############################################
#                   모델링                     #
##############################################
# ****************** 연속형 ****************** #
########### 1. 선형회귀 
from sklearn.linear_model import LinearRegression
# 모델 만들기 
model = LinearRegression()
# 학습 데이터로 모델 학습 
model.fit(x_train, y_train)
# 학습된 모델로 값 예측 
y_train_predicted = model.predict(x_train)
y_test_predicted = model.predict(x_test)
# 선형회귀 모델로 도출된 y 절편 구하기
print("선형회귀 모델로 도출된 y 절편 구하기")
print(model.intercept_)
# 선형회귀 모델에 포함된 독립변수들의 각 기울기 값 구하기
print("회귀계수 값 ")
print(model.coef_) # 회귀계수 값 
# 선형회귀 분석의 예측모델에는 MAE MSE RMSE 결정계수 등의 지표로 평가 수행 
print("학습 데이터에 대한 결정계수")
print(model.score(x_train, y_train))
print("테스트 데이터에 대한 결정계수")
print(model.score(x_test, y_test))

# 평가하기
# 결정계수를 계산
from sklearn.metrics import r2_score
# MAE를 계산
from sklearn.metrics import mean_absolute_error
# MSE를 계산
from sklearn.metrics import mean_squared_error
# 제곱근 계산
import numpy as np 

print("\n선형회귀")
# 학습데이터의 결정계수 구하기
print("학습데이터의 결정계수 구하기")
print(r2_score(y_train, y_train_predicted))
# 테스트데이터의 결정계수 구하기
print("# 테스트데이터의 결정계수 구하기")
print(r2_score(y_test, y_test_predicted))
# 테스트데이터의 MSE 지표 구하기 
print("테스트데이터의 MSE 지표 구하기 ")
print(mean_squared_error(y_test, y_test_predicted))
# 테스트데이터의 RMSE 지표 구하기 
print("테스트데이터의 RMSE 지표 구하기 ")
print(np.sqrt(mean_squared_error(y_test, y_test_predicted)))
# 테스트데이터의 MAE 지표 구하기
print("테스트데이터의 MAE 지표 구하기")
print(mean_absolute_error(y_test, y_test_predicted))

########### 2. 랜덤 포레스트 회귀 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_train_predicted = model.predict(x_train)
y_test_predicted = model.predict(x_test)

print("\n랜덤포레스트")
# 학습데이터의 결정계수 구하기
print("학습데이터의 결정계수 구하기")
print(r2_score(y_train, y_train_predicted))
# 테스트데이터의 결정계수 구하기
print("# 테스트데이터의 결정계수 구하기")
print(r2_score(y_test, y_test_predicted))
# 테스트데이터의 MSE 지표 구하기 
print("테스트데이터의 MSE 지표 구하기 ")
print(mean_squared_error(y_test, y_test_predicted))
# 테스트데이터의 RMSE 지표 구하기 
print("테스트데이터의 RMSE 지표 구하기 ")
print(np.sqrt(mean_squared_error(y_test, y_test_predicted)))
# 테스트데이터의 MAE 지표 구하기
print("테스트데이터의 MAE 지표 구하기")
print(mean_absolute_error(y_test, y_test_predicted))

# ***** 하이퍼파라미터 튜닝 ***** #
# n_estimators : 트리의 개수, criterion : 트리를 분할하는 기준 
# 아래 모델은 MAE 평가지표를 향상시키는 방법임 
print("\n하이퍼파라미터 튜닝으로 MAE 평가지표 향상. MAE는 0에 수렴할수록 성능 좋아짐")
model = RandomForestRegressor(n_estimators=1000, criterion='absolute_error')
model.fit(x_train, y_train)
y_train_predicted = model.predict(x_train)
y_test_predicted = model.predict(x_test)

print("학습데이터의 결정계수 구하기")
print(r2_score(y_train, y_train_predicted))
# 테스트데이터의 결정계수 구하기
print("# 테스트데이터의 결정계수 구하기")
print(r2_score(y_test, y_test_predicted))
# 테스트데이터의 MSE 지표 구하기 
print("테스트데이터의 MSE 지표 구하기 ")
print(mean_squared_error(y_test, y_test_predicted))
# 테스트데이터의 RMSE 지표 구하기 
print("테스트데이터의 RMSE 지표 구하기 ")
print(np.sqrt(mean_squared_error(y_test, y_test_predicted)))
# 테스트데이터의 MAE 지표 구하기
print("테스트데이터의 MAE 지표 구하기")
print(mean_absolute_error(y_test, y_test_predicted))


########### 3. 그레디언트 부스팅 회귀 
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=10)
model.fit(x_train, y_train)
y_train_predicted = model.predict(x_train)
y_test_predicted = model.predict(x_test)

print("\n그레디언트 부스팅 회귀 ")
# 학습데이터의 결정계수 구하기
print("학습데이터의 결정계수 구하기")
print(r2_score(y_train, y_train_predicted))
# 테스트데이터의 결정계수 구하기
print("# 테스트데이터의 결정계수 구하기")
print(r2_score(y_test, y_test_predicted))
# 테스트데이터의 MSE 지표 구하기 
print("테스트데이터의 MSE 지표 구하기 ")
print(mean_squared_error(y_test, y_test_predicted))
# 테스트데이터의 RMSE 지표 구하기 
print("테스트데이터의 RMSE 지표 구하기 ")
print(np.sqrt(mean_squared_error(y_test, y_test_predicted)))
# 테스트데이터의 MAE 지표 구하기
print("테스트데이터의 MAE 지표 구하기")
print(mean_absolute_error(y_test, y_test_predicted))


########### 4. XGB 부스팅 회귀 
from xgboost import XGBRegressor
model = XGBRegressor(random_state=10)
model.fit(x_train, y_train)
y_train_predicted = model.predict(x_train)
y_test_predicted = model.predict(x_test)

print("\nXGB 부스팅 회귀 ")
# 학습데이터의 결정계수 구하기
print("학습데이터의 결정계수 구하기")
print(r2_score(y_train, y_train_predicted))
# 테스트데이터의 결정계수 구하기
print("# 테스트데이터의 결정계수 구하기")
print(r2_score(y_test, y_test_predicted))
# 테스트데이터의 MSE 지표 구하기 
print("테스트데이터의 MSE 지표 구하기 ")
print(mean_squared_error(y_test, y_test_predicted))
# 테스트데이터의 RMSE 지표 구하기 
print("테스트데이터의 RMSE 지표 구하기 ")
print(np.sqrt(mean_squared_error(y_test, y_test_predicted)))
# 테스트데이터의 MAE 지표 구하기
print("테스트데이터의 MAE 지표 구하기")
print(mean_absolute_error(y_test, y_test_predicted))



# ****************** 범주형 ****************** #
# 분류 모델링 
x_train2 = x_train.drop(columns='am_new')
y_train2 = x_train['am_new']
x_test2 = x_test.drop(columns='am_new')
y_test2 = x_test['am_new']

########### 1. 의사결정나무 분류 
print("\n의사결정나무 분류")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train2, y_train2)
y_test2_predicted = model.predict(x_test2)
print(y_test2_predicted)

# 평가하기
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
# 테스트 데이터의 ROC-AUC 구하기
print("테스트 데이터의 ROC-AUC 구하기")
print(roc_auc_score(y_test2, y_test2_predicted))
print("테스트 데이터의 정확도 구하기")
print(accuracy_score(y_test2, y_test2_predicted))
print("테스트 데이터의 정밀도 구하기")
print(precision_score(y_test2, y_test2_predicted))
print("테스트 데이터의 재현율 구하기")
print(recall_score(y_test2, y_test2_predicted))


########### 2. 랜덤포레스트 분류 
print("\n랜덤포레스트 분류")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train2, y_train2)
y_test2_predicted = model.predict(x_test2)
print(y_test2_predicted)

# 테스트 데이터의 ROC-AUC 구하기
print("테스트 데이터의 ROC-AUC 구하기")
print(roc_auc_score(y_test2, y_test2_predicted))


########### 3. 로지스틱회귀 분류 
print("\n로지스틱회귀 분류")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train2, y_train2)
y_test2_predicted = model.predict(x_test2)
print(y_test2_predicted)

# 테스트 데이터의 ROC-AUC 구하기
print("테스트 데이터의 ROC-AUC 구하기")
print(roc_auc_score(y_test2, y_test2_predicted))



########### 4. 익스트림 그레디언트 부스팅 분류 
print("\n익스트림 그레디언트 부스팅 분류")
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train2, y_train2)
y_test2_predicted = model.predict(x_test2)
print(y_test2_predicted)

# 테스트 데이터의 ROC-AUC 구하기
print("테스트 데이터의 ROC-AUC 구하기")
print(roc_auc_score(y_test2, y_test2_predicted))


# 종속변수의 0과 1에 대한 확률값을 계산해주는 함수 
y_test2_proba = model.predict_proba(x_train2)
print("\n종속변수의 0과 1에 대한 확률값을 계산해주는 함수 ")
# 1열은 0 값으로 분류될 확률, 2열은 1값으로 분류될 확률
print(y_test2_proba)


##############################################
#                  저장하기                    #
##############################################
# 제출할 y_test2_predicted 변수의 데이터 타입 확인
print(type(y_test2_predicted))
# 제출할 변수를 데이터 프레임으로 변경 후 data 디렉터리 하위에 csv 파일 저장
# index는 명시가 없으면 반드시 false로 지정해주기!!
pd.DataFrame(y_test2_predicted).to_csv('./data/수험번호.csv', index = False)