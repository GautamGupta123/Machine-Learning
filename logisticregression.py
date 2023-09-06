import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
house=pd.read_csv("csv/USA_housing.csv")
house.head(5)
house.drop(['Address'],axis=1)
house.shape
house.describe()
x=house.iloc[:,[2,3]].values
y=house.iloc[:,4].values
print(x)
house['Address']=house['Address'].astype('category')
house['Address']=house['Address'].cat.codes
print(y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)
print(xtrain[0:10,:])
from sklearn.linear_model import LogisticRegression
  
c = LogisticRegression(random_state = 0)
c.fit(xtest,ytrain)
pred = c.predict(xtest)
from sklearn.metrics import confusion_matrix
  
confusem = confusion_matrix(ytest, pred)
print ("Confusion Matrix : \n", confusem)
from sklearn.metrics import accuracy_score
  
print ("Accuracy : ", accuracy_score(ytest, pred))