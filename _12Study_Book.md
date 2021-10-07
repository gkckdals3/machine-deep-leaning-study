# machine-deep-leaning-study

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine=pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()

wine_data=wine[['alcohol','sugar','pH']]
wine_target=wine['class']
train_input,test_input,train_target,test_target=train_test_split(wine_data,wine_target,test_size=0.2,random_state=42)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=-1,random_state=42)

print(np.mean(scores['train_score']),np.mean(scores['test_score']))

rf.fit(train_input,train_target)
print(rf.feature_importances_) # DecisionTreeClassifier가 제공하는 중요한
#매개변수를 모두 제공
#특성중요도 또한 계산 가능

#OOB샘플 : 부트스트랩 샘플에 포함되지 않고 남는 샘플
#부트스트랩 샘플로 훈련한 결정트리를 평가할 수 있다.
rf=RandomForestClassifier(oob_score=True,n_jobs=-1,random_state=42)
rf.fit(train_input,train_target)
print(rf.oob_score_)

#엑스트라 트리 : 랜덤 포레스트와 매우 비슷하게 동작
#기본적으로 100개의 결정트리를 훈련
#랜덤 포레스트와 엑스트라 트리의 차이점은 부트스트랩 샘플을 사용하지 않는다는 점
#즉 각 결정 트리를 만들 때 전체 훈련 세트를 사용
#하나의 결정 트리에서 특성을 무작위로 분할한다면 성능이 낮아지겠지만 많은 트리를
#앙상블 하기 때문에 과대적합을 막고 검증 세트의 점수를 높일 수 있다.
from sklearn.ensemble import ExtraTreesClassifier
et= ExtraTreesClassifier(n_jobs=-1,random_state=42)
scores=cross_validate(et,train_input,train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score'],np.mean(scores['test_score'])))
#엑스트라 트리가 무작위성이 좀 더 크기 때문에 랜덤 포레스트보다 더 많은
#결정 트리를 훈련해야 한다.
#하지만 랜덤하게 노드를 분할하기 때문에 빠른 계산 속다가 엑스트라 트리의 장점이다.

#그레이디언트 부스팅
#깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블
#하는 방법
#기본적으로 깊이가 3인 결정 트리를 100개 사용

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
gb=GradientBoostingClassifier(random_state=42)
scores=cross_validate(gb,train_input, train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))

gb=gradientBoostingClassifier(n_estimator=500,learning_rate=0.2,random_state=42)
scores=cross_validate(gb,train_input,train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))

gb.fit(train_input,train_target)
print(gb.feature_importances_)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb=HistGradientBoostingClassifier(random_state=42)
scores=cross_validate(hgb,train_input,train_target,return_train_score=True)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))

from sklearn.inspection import permutation_importance
hgb.fit(train_input,train_target)
result=permutation_importance(hgb,train_input,train_target,n_repeats=10,random_state=42,n_jobs=-1)
print(result.importances_mean)

result=permutation_importance(hgb,test_input,test_target,n_repeats=10,random_state=42,n_jobs=-1)
print(result.importances_mean)

hgb.score(test_input,test_target)

!pip3 install cmake

!pip3 install xgboost

from xgboost import XGBClassifier
xgb=XGBClassifier(tree_method='hist',random_state=42)
score=cross_validate(xgb,train_input,train_target,return_train_score=True)
print(np.mean(score['train_score']),np.mean(score['test_score']))

