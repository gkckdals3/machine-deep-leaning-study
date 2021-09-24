# machine-deep-leaning-study

import pandas as pd
df=pd.read_csv('https://bit.ly/perch_csv_data')
perch_full=df.to_numpy()
print(perch_full[0:5])

import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
       
       from sklearn.preprocessing import PolynomialFeatures#사이킷 런 전처리 패키지
poly=PolynomialFeatures()
poly.fit([[2,3]])
poly.transform([[2,3]])

poly=PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
poly.transform([[2,3]])


poly=PolynomialFeatures(include_bias=0)
poly.fit(train_input)
train_poly=poly.transform(train_input)
print(train_poly.shape)

poly.get_feature_names()# X0^2는 첫번째 특성의 제곱 X0X1은 첫번째 특성과 두번째 특성의 곱

test_poly=poly.transform(test_input)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=5,include_bias=0)
poly.fit(train_input)
train_poly=poly.transform(train_input)
test_poly=poly.transform(test_input)
print(train_poly.shape)

lr.fit(train_poly,train_target)
lr.score(train_poly,train_target)

lr.score(test_poly,test_target)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train_poly)
train_scaled=ss.transform(train_poly)
test_scaled=ss.transform(test_poly)

from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

#alpha값을 찾는 방법1
train_score= []
test_score=[]
import matplotlib.pyplot as plt
alpha_list=[0.001,0.01,0.1,1,10,100,100]
for alpha in alpha_list:
    #릿지 모델 만들기
    ridge=Ridge(alpha=alpha)
    #릿지 모델 훈련
    ridge.fit(train_scaled,train_target)
    train_score.append(ridge.score(train_scaled,train_target))
    test_score.append(ridge.score(test_scaled,test_target))
    
    plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)#alpha값을 0,001부터 10배씩늘렸기 때문에
#왼쪽이 너무촘촘한 형태이므로 로그를 취함으로써 6개의 값을 동일한 간격으로 나타낸다.
'''그래프에서 확인했을때 훈련세트에는 잘 맞고 테스트 세트에는 형편없는 과대적합의
전형적인 모습이다. 오른쪽으로 갈수록 둘다 성능이 떨어지는 과소적합의 형태가 되는것
을 확인할 수 있다. 그러므로 가장 두그래프가 가까우면서 테스트 세트의 점수가
가장 높은 -1 을 선택하여 alpha값을 0.1로한 모델을 최종 모델로 선정하였다.'''

ridge=Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))#규제를 위한 릿지회귀 끝

#라쏘 회귀
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))

train_score=[]
test_score=[]
alpha_list=[0.001,0.01,0.1,1,10,100]
for alpha in alpha_list:
    lasso=Lasso(alpha=alpha,max_iter=10000)
    lasso.fit(train_scaled,train_target)
    train_score.append(lasso.score(train_scaled,train_target))
    test_score.append(lasso.score(test_scaled,test_target))

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')

lasso=Lasso(alpha=10)
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))

print(np.sum(lasso.coef_==0))#15개의 특성만 사용한 것을 알 수 있다.
#특성 공학은 주어진 특성을 조합하여 새로운 특성을 만드는 일련의 작업과정
#릿지는 규제가 있는 선형 회귀 모델 중 하나이며 선형 모델의 계수를 작게 만들어 과대적합을 완화
#라쏘는 또 다른 규제가 있는 성형 회귀 모델이다. 릿지와 달리 계수 값을 아예 0으로 만들 수 있다.

