import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
# import numpy as np

data=pd.read_csv("csv/heart.csv")
print(data.tail())

print(data.describe())
print(data.shape)
print(data.columns)

X=data.drop('target',axis=1)
Y=data['target']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,random_state=42)
X_test.values

from sklearn.ensemble import RandomForestClassifier
classifier_random=RandomForestClassifier(random_state=42,n_jobs=-1,max_depth=5,n_estimators=100,oob_score=True)

classifier_random.fit(X_train, Y_train)

classifier_random.oob_score_

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")


rf_best = grid_search.estimator
from sklearn.tree import plot_tree
plt.figure(figsize=(80,40))
plot_tree(rf_best.estimators[2],feature_names=X.columns,class_names=['Disease',"No Disease"],filled=True)




