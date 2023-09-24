import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("csv/heart.csv")

X=data.iloc[:,0:-1].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from LRS import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)
y_pred=regression.predict(X_test)

plt.scatter(X,y,color='black')
plt.plot(X_test,regression.predict(X_test))
plt.show()