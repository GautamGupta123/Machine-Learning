import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("csv/heart.csv")
X=data.iloc[:5,0:2]
print(X)
print(data.head(5))

plt.scatter(data.age,data.sex,marker='+',color='red')
plt.show()

from sklearn.model_selection import train_test_split
xtrain, xtest ,ytrain, ytest =  train_test_split (data.age,data.sex,test_size=0.3)

print(xtest)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,ytrain)
model.predict(xtest,ytrain)
model.score(xtest,ytest)