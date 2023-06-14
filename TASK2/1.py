# 문제 : 고객 3,500명에 대한 학습용 데이터(x_train.csv, y_train.csv)를 이용하여 성별 예측 모형을 만든 후, 
# 이를 평가용 데이터(x_test.csv)에 적용하여 얻은 2,482명 고객의 성별 예측값을 다음과 같은 형식의 csv 파일로 생성하셈 
# 모델의 성능은 ROC-AUC 평가지표에 땨라 매겨짐
# 제출 형식
# custid, gender
# 3500, 0.267
# 3501, 0.578


##############################################
#                 데이터 탐색                   #
##############################################
# 1. 데이터 가져오기
import pandas as pd

print("데이터 가져오기")
x_train = pd.read_csv('./bigData-main/x_train.csv', encoding='CP949')
x_test = pd.read_csv('./bigData-main/x_test.csv', encoding='CP949')
y_train = pd.read_csv('./bigData-main/y_train.csv', encoding='CP949')
print(x_train.head())
print(x_test.head())
print(y_train.head())


# 2. 행과 열을 바꾸어 보기
print("\n행과 열을 바꾸어 보기")
print(x_train.head().T)
print(x_test.head().T)
print(y_train.head().T)

# 3. 행열 확인하기
print("\n행열 확인하기")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

# 4. 요약정보 확인
print("\n요약정보 확인")
print(x_train.info())

# 5. 기초통계량 확인하기 
print("\n기초통계량 확인하기")
print(x_train.describe().T)

##############################################
#                데이터 전처리                  #
##############################################
# 테스트 데이터인 x_test도 값을 예측하는 과정에 사용하므로 전처리 필요

# 1. 불필요한 컬럼 삭제 
# cust_id 칼럼은 종속병수인 성별을 예측하는 정보가 아니라 key 역할이므로 삭제
# 단, 최종 제출에는 사용되는 컬럼이므로 따로 저장
x_test_cust_id = x_test['cust_id']

# cust_id 삭제
x_train = x_train.drop(columns = ['cust_id'])
x_test = x_test.drop(columns = ['cust_id'])
y_train = y_train.drop(columns = ['cust_id'])

# 컬럼이 삭제된 상위 5개 행 확인
print(x_train.head())
print(x_test.head())
print(y_train.head())

# 2. 결측치 처리
# 환불금액의 칼람에 2,295건의 결측치 존재
print(x_train.isnull().sum())
# 이는 환불금액 결측치는 이력이 없는 경우에 발생할 것으로 예상할 수 있음 -> 0으로 대체
x_train['환불금액'] = x_train['환불금액'].fillna(0)
x_test['환불금액'] = x_test['환불금액'].fillna(0)
# 결측치가 조치되었는지 확인
print(x_train['환불금액'].isnull().sum())
print(x_test['환불금액'].isnull().sum())

# 3. 범주형 변수를 인코딩하기 
# 주구매상품 칼럼에서 중복을 제외한 값들을 확인
print(x_train['주구매상품'].unique())
# 주구매지점 칼럼에서 중복을 제외한 값들을 확인
print(x_train['주구매지점'].unique())
# 주구매지점 칼럼에서 중복을 제외한 값들의 개수 세기 
print(x_train['주구매지점'].unique().size)
# 인코딩할 수가 많을 경우, 라벨 인코딩을 하는 것이 효과적임

from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()
# 주구매상품에 대해 라벨 인코딩을 수행하고, 주구매상품 칼람으로 다시 저장
x_train['주구매상품'] = encoder.fit_transform(x_train['주구매상품'])
# 라벨 인코딩 결과를 호가인하기 위해, 상위 10개 행을 확인
print(x_train['주구매상품'].head(10))
# 주구매상품 컬럼에 대한 라벨 인코딩의 변환 순서 확인
print(encoder.classes_)
# 테스트 데이터도 라벨 인코딩 수행
x_test['주구매상품'] = encoder.fit_transform(x_test['주구매상품'])

# 주구매지점 라벨 인코딩 수행 
x_train['주구매지점'] = encoder.fit_transform(x_train['주구매지점'])
# 라벨 인코딩 결과를 호가인하기 위해, 상위 10개 행을 확인
print(x_train['주구매지점'].head(10))
# 주구매지점 컬럼에 대한 라벨 인코딩의 변환 순서 확인
print(encoder.classes_)
# 테스트 데이터도 라벨 인코딩 수행
x_test['주구매지점'] = encoder.fit_transform(x_test['주구매지점'])

# 4. 파생변수 만들기
# 환불금액이 0보다 크면 1, 0과 같으면 0 (환불여부)
condition = x_train['환불금액'] > 0
# 조건에 맞으면 1
x_train.loc[condition, '환불금액_new'] = 1
# 조건에 안맞으면 0
x_train.loc[~condition, '환불금액_new'] = 0
# 확인
print(x_train[['환불금액', '환불금액_new']])
# 기존의 환불금액 컬럼 삭제
x_train = x_train.drop(columns=['환불금액'])

# 테스트 데이터도 같은 과정 거치기 
condition = x_test['환불금액'] > 0
# 조건에 맞으면 1
x_test.loc[condition, '환불금액_new'] = 1
# 조건에 안맞으면 0
x_test.loc[~condition, '환불금액_new'] = 0
# 확인
print(x_test[['환불금액', '환불금액_new']])
# 기존의 환불금액 컬럼 삭제
x_test = x_test.drop(columns=['환불금액'])

# 5. 표준화 크기로 변환 
# 크기변환 전, x_train 세트의 기초 통계량 확인
print(x_train.describe().T)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 표준크기변환 수행 후 x_train 칼럼명 그래도 사용
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
# 테스트 데이터도 크기변환 
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
# 크기변환 후, x_train 세트의 기초 통계량 확인
print(x_train.describe().T)

# 6. 상관관계 확인하기 
# 총구매액, 최대구매액, 환불금액_new 칼람관의 상관관계 구하기
print(x_train[['총구매액', '최대구매액', '환불금액_new']].corr())
# 일반적으로 상관관계가 0.6 이상이면 강한 상관관계가 존재한다고 해석 
# 따라서 총구매액과 최대구매액 유사성 높음 > 따라서 둘 중 하나는 다중공선성을 이유로 삭제
# 최대구매액 삭제 
x_train = x_train.drop(columns=['최대구매액'])
# 테스트 세트에서도 삭제
x_test = x_test.drop(columns=['최대구매액'])

##############################################
#                학습 및 평가                   #
##############################################
# 1. 데이터 학습시키기 
# 의사결정나무 분류기 활용 
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# x_train, y_train 세트로 model을 학습시키기
model.fit(x_train, y_train)
# 학습된 model을 활용해 테스트 데이터의 종속변수를 예측
y_test_predicted = model.predict(x_test)
# 예측한 결과 출력
print(pd.DataFrame(y_test_predicted))

# 2. 하이퍼 파라미터 튜닝 
# 의사결정나무분류기의 대표적인 하이퍼 파라미터 max_depth(깊이제한), criterion(트리노드분기)
model = DecisionTreeClassifier(max_depth=10, criterion='entropy', random_state=10)
# x_train, y_train 세트로 model을 학습시키기
model.fit(x_train, y_train)
# 학습된 model을 활용해 테스트 데이터의 종속변수를 예측
y_test_predicted = model.predict(x_test)
# 예측한 결과 출력
print(pd.DataFrame(y_test_predicted))
# 근데 시험에서는 걍 하이퍼파라미터 입력하지 않은채 데이터를 학습시켜도 합격하는 수준에 무리 없음!

# 3. 결과 예측하기 
# model을 통해 x_test에 맞는 종속변수 확률 구하기 
y_test_proba = model.predict_proba(x_test)
# 종속 변수의 0, 1에 대한 각 확률을 확인하기
print(pd.DataFrame(y_test_proba).head())
# 최종적으로는 남자를 구해야함
print(pd.DataFrame(y_test_proba)[1])
result = pd.DataFrame(y_test_proba)[1]

# 4. 모델 평가하기
# x_train에 대한 종속변수를 예측
y_train_predicted = model.predict(x_train)
# roc 평가지표를 계산하기 위한 함수를 가져오기
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train, y_train_predicted))

##############################################
#                 결과 제출                    #
##############################################

# x_test_cust_id 변수과 result 변수를 세로 방향으로 붙이기
pd.concat([x_test_cust_id, result], axis=1)
print(pd.concat([x_test_cust_id, result], axis=1))

# 1 컬럼명을 gender 컬럼명으로 변경하여 다시 결과 확인
pd.concat([x_test_cust_id, result], axis=1).rename(columns={1:'gender'})
print(pd.concat([x_test_cust_id, result], axis=1).rename(columns={1:'gender'}))

# 앞의 출력 결과를 저장
final = pd.concat([x_test_cust_id, result], axis=1).rename(columns={1:'gender'})

# 파일로 저장
final.to_csv('data/21800491.csv', index = False)