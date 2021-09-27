# machine-deep-leaning-study
import pandas as pd
fish=pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()

print(pd.unique(fish['Species']))

fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[0:5])
fish_target=fish[['Species']].to_numpy()

print(fish_target)

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target,random_state=42)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

print(train_scaled)

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled,train_target)
print(kn.score(train_scaled,train_target))
print(kn.score(test_scaled,test_target))

print(kn.classes_)

print(kn.predict(test_scaled[0:5]))

import numpy as np
proba=kn.predict_proba(test_scaled[0:5])
print(np.round(proba,decimals=4))

distances,indexes=kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

char_arr=np.array(['A','B','C','D','E'])
print(char_arr[[True,False,True,False,False]])
char_arr

bream_smelt_indexes=np.array((train_target=='Bream')|(train_target=='Smelt'))
train_bream_smelt=train_scaled[bream_smelt_indexes]
target_bream_smelt=train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)

lr=LogisticRegression(C=20,max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

print(lr.predict(test_scaled[0:5]))

proba=lr.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=3))

print(lr.classes_)

from sklearn.linear_model import LogisticRegression
decision=lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2))

from scipy.special import softmax
proba=softmax(decision, axis=1)
print(np.round(proba,decimals=3))
