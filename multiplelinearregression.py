import pandas as pd
import numpy as np
house=pd.read_csv("csv/USA_Housing.csv")
print(house)
house['Address']=house['Address'].astype('category')
house['Address']=house['Address'].cat.codes
house.iloc[1:2,:2].values

#tochecknull values
house.isnull().sum()
X=house.drop(columns='Address')
print(X)
y=house['Address']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
predict=lr.predict(X_train)
print(predict)
import matplotlib.pyplot as plt
plt.scatter(y_train,predict)
plt.show()
from sklearn.metrics import r2_score
lr.score(X_train,y_train)
lr.score(X_test,y_test)

