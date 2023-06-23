# Q. [마케팅] 자동차 시장 세분화
# 자동차 회사는 새로운 전략을 수립하기 위해 4개의 시장으로 세분화했습니다.
# 기존 고객 분류 자료를 바탕으로 신규 고객이 어떤 분류에 속할지 예측해주세요!
# 예측할 값(y): "Segmentation" (1,2,3,4)

# 평가: Macro f1-score
# data: train.csv, test.csv
# 제출 형식:
# ID,Segmentation
# 458989,1
# 458994,2
# 459000,3
# 459003,4

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'ID': test.ID, 'Segmentation': pred}).to_csv('003000000.csv', index=False)
# 노트북 구분¶
# basic: 수치형 데이터만 활용 -> 학습 및 test데이터 예측
# intermediate: 범주형 데이터도 활용 -> 학습 및 test데이터 예측
# advanced: 학습 및 교차 검증(모델 평가) -> 하이퍼파라미터 튜닝 -> test데이터 예측
# 학습을 위한 채점
# 최종 파일을 "수험번호.csv"가 아닌 "submission.csv" 작성 후 오른쪽 메뉴 아래 "submit" 버튼 클릭 -> 리더보드에 점수 및 등수 확인 가능함
# pd.DataFrame({'ID': test.ID, 'Segmentation': pred}).to_csv('submission.csv', index=False)

import pandas as pd
train = pd.read_csv('past/data/test_4.csv')
print(train.head(1).T)
print(train['Segmentation'].value_counts())


# target(y, label) 값 복사
target = train.pop('Segmentation')
target

# test데이터 ID 복사
test_ID = test.pop('ID')

# 수치형 컬럼(train)
# ['ID', 'Age', 'Work_Experience', 'Family_Size', 'Segmentation']
num_cols = ['Age', 'Work_Experience', 'Family_Size']
train = train[num_cols]
train.head(2)

# 수치형 컬럼(test)
test = test[num_cols]
test.head(2)

# 모델 선택 및 학습
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)
rf.fit(train, target)
pred = rf.predict(test)
pred

# 예측 결과 -> 데이터 프레임
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)

submit = pd.DataFrame({
    'ID': test_ID,
    'Segmentation': pred
})
submit

submit.to_csv("submission.csv", index=False)
# Score: 0.30477

















# f5컬럼을 기준으로 내림차순 정렬
df = df.sort_values('f5', ascending=False)
df.head(10)



# 최소값 찾기
min = df['f5'][:10].min()
# min = 91.297791
min

df.iloc[:10,-1] = min
df.head(10)

# 80세 이상의 f5컬럼 평균
print(df[df['age']>=80]['f5'].mean())



# 데이터 나누기 방법1
data70 = df.iloc[:70]
data30 = df.iloc[70:]

# [심화학습] 데이터 나누기 방법2
# data70, data30 = np.split(df, [int(.7*len(df))])

# [심화학습] 데이터 나누기 방법3 (랜덤으로 샘플링하라고 했을 때!!)
# data70 = df.sample(frac = 0.7)
# data70 = df.drop(data70.index)

data70.tail()
## 결측치 확인
data70.isnull().sum()
## 결측치 채우기 전 f1컬럼 표준편자
std1 = data70['f1'].std()
std1
## 중앙값 확인
med=data70['f1'].median()
med
## 중앙값으로 채우기
data70['f1'] = data70['f1'].fillna(med)

## 다른 방법들
# data70['f1']= data70['f1'].replace(np.nan, med)
# data70 = data70.fillna(value=med)
## 결측치 확인
data70.isnull().sum()
## 결측치를 채운 후 표준편차 구하기
std2 = data70['f1'].std()
std2
print(std1-std2)

3.2965018033960725

std = df['age'].std() * 1.5
mean = df['age'].mean()

min_out = mean - std
max_out = mean + std
print(min_out, max_out)

# 이상치 age합
df[(df['age']>max_out)|(df['age']<min_out)]['age'].sum()

# 다르게 작성방법
# df.loc[(df['age'] > max)]['age'].sum() + df.loc[(df['age']< min)]['age'].sum()