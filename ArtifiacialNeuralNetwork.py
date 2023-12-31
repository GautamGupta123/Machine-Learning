import pandas as pd
import numpy as np

data = pd.read_csv("csv/Churn_Modelling.csv")
X = data.iloc[:,3:-1].values
print(X)

Y = data.iloc[:,-1].values

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

LE1 = LabelEncoder()
X[:,2] = np.array(LE1.fit_transform(X[:,2]))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs = 100)
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])) > 0.5)
