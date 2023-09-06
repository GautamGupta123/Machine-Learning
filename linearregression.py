import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
house=pd.read_csv("csv/USA_Housing.csv")
house.head()
print(house.shape)
house.describe()
print(house.columns)
X=house[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
           'Avg. Area Number of Bedrooms', 'Area Population', 'Price']]
y=house['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lm=LinearRegression()
print(X_train)
lm.fit(X_train,y_train)
coeff=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficent'])
print(coeff)
