import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("csv/Social_Network_Ads.csv");
print(data.head())

print(data.shape)
print(data.columns)

X=data.iloc[:,[2,3]]
Y=data.iloc[:,4]

print(X.head())
print(Y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

print("Training data:",X_train.shape)
print("Testing Data", X_test.shape)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

print(Y_pred)

from sklearn import metrics
print('Accuracy Score:')
print(metrics.accuracy_score(Y_test,Y_pred))

plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Test Data')
plt.show()

plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)

w=classifier.coef_[0]
a=-w[0]/w[1]

xx=np.linspace(-2.5,2.5)
yy=a*xx- (classifier.intercept_[0]) / w[1]

plt.plot(xx,yy)
plt.axis("off")
plt.show()