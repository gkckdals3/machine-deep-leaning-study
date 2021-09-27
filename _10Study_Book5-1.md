# machine-deep-leaning-study
import pandas as pd
wine=pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()

wine.info()

wine.describe()
#1사분위수=25%값
#2사분위수=50%값
#3사분위수=75%값

wine_data=wine[['alcohol','sugar','pH']].to_numpy()
wine_target=wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(wine_data,wine_target,test_size=0.2,random_state=42)

print(train_input.shape,test_input.shape)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled,train_target)
print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt,max_depth=1,filled=True,feature_names=['alcohol','sugar','pH'])
plt.show()

dt=DecisionTreeClassifier(max_depth=3,random_state=42)
dt.fit(train_scaled,train_target)
print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))

from sklearn.tree import plot_tree
plt.figure(figsize=(20,15))
plot_tree(dt,filled=True,feature_names=['alcohol','sugar','pH'])
plt.show()
