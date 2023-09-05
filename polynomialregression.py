import pandas as pd
house=pd.read_csv("USA_Housing.csv")
print(house)
house.head()
house.isna().sum()

house['Address']=house['Address'].astype('category')
house['Address']=house['Address'].cat.codes


X=house.drop(columns=['Area Population'])
y=house['Area Population']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40,random_state=1)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=7)
X_poly_train=poly.fit_transform(X_train_scaler)
X_test_poly=poly.transform(X_test_scaler)
poly.fit(X_poly_train,y_train)
lr.fit(X_poly_train,y_train)
y_pred=lr.predict(X_test_poly)

print(y_pred)

import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)

import seaborn as sns
sns.distplot((y_train),bins=50)



