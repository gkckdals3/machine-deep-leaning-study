# machine-deep-leaning-study

import pandas as pd
wine=pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()

wine_data=wine[['alcohol','sugar','pH']].to_numpy()
wine_target=wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
#테스트 데이터셋
train_input,test_input,train_target,test_target=train_test_split(wine_data,wine_target,test_size=0.2,random_state=42)

#검증 데이터셋
sub_input,val_input,sub_target,val_target=train_test_split(train_input,train_target,test_size=0.2,random_state=42)

print(sub_input.shape,val_input.shape)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(sub_input,sub_target)
print(dt.score(sub_input,sub_target))
print(dt.score(val_input,val_target))
#훈련 세트에 과대적합

#교차 검증을 이용하면 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터의 사용이 가능
#검증 세트를 떼어 내어 평가화는 과정을 반복한다.
from sklearn.model_selection import cross_validate
scores=cross_validate(dt,train_input,train_target)
print(scores)
#처음 2개의 키는 각각 모델을 훈련하는 시간과 검증하는 시간을 의미한다.

#교차 점수의 최종 점수는 scores['test_score']에 담긴 5개의 점수를 평균하여
#얻을 수 있다.
import numpy as np
np.mean(scores['test_score'])

#사이킷런의 분할기
#교차 검증을 할 때 훈련 세트를 섞기위해 사용
#creoss_validate()함수는 기본적으로 회귀 모델일 경우 KFold 분할기를 사용하고
#분류모델의 경우에는 StratifiedKFold를 사용한다.
from sklearn.model_selection import StratifiedKFold
scores=cross_validate(dt,train_input,train_target,cv=StratifiedKFold())
np.mean(scores['test_score'])

from sklearn.model_selection import GridSearchCV
params={'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}
#min_impurity_decrese: 분할로 얻어질 최소한의 불순도

gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
#n_jobs=-1로 설정하면 컴퓨터의 모든 코어를 사용한다.
print(type(train_input))
print(type(train_target))
print(type(gs))

gs.fit(train_input,train_target)

dt=gs.best_estimator_
print(dt.score(train_input,train_target))

print(gs.best_params)#최상의 매개변수

print(np.max(gs.cv_results_['mean_test_score']))#최상의 교차 검증 점수


from scipy.stats import uniform, randint

rgen=randint(0,10)
rgen.rvs(10)# uniform과 randint 클래스는 모두 주어진 범위에서 고르게 값을 뽑는다.
#이를 균등 분포에서 샘플린 한다고 한다.
#randint=정수값을 뽑고 uniform은 실숫값을 뽑는다.

np.unique(rgen.rvs(1000),return_counts=True)#return_counts는 각 숫자의 개수를 센다.

ugen=uniform(0,10)
ugen.rvs(10)

params={'min_impurity_decrease':uniform(0.0001,0.001),
        'max_depth':randint(20,50),
        'min_samples_split' : randint(2,25),
        'min)_sampeles_leaf': randint(1,25),}
#샘플링의 횟수는 사이킷런의 랜덤 서치 클래스인 RandomizedSearchCV의 n_iter의
#매계변수에 지정한다.

from sklearn.model_selection import RandomizedSearchCV
gs=RandomizedSearchCV(DecisionTreeClassifier(random_state=42),params,n_iter=100,n_jobs=-1,random_state=42)
gs.fit(train_input,train_target)

print(gs.best_params_)#최적의 매개변수 조합

print(np.max(gs.cv_results_['mean_test_score']))#최고의 교차 검증 점수

dt=gs.best_estimator_#최적의 모델은 이미 전체 훈련세트로 훈련되어 
#best_esimator_ 속성에 저장되어 있다.
print(dt.score(test_input,test_target))

#정리
#검증 세트는 하이퍼파라미터 튜닝을 위해 모델을 평가할 때, 테스트 세트를 사용하지
#않기 위해 사용한다.

#교차 검증은 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고
#나머지 폴드에서는 모델을 훈련한다. 교차 검증은 이런 식으로 모든 폴드에 대해
#검증 점수를 얻어 평균하는 방법이다.

#그리드 서치는 하이퍼파라미터 탐색을 자동화 해주는 도구이다.

#랜덤 서치는 연속된 매개변수 값을 탐색할 때 유용하다.

