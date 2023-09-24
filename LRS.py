import numpy as np
class LinearRegression:
    def __init__(self, learning_rate=0.1,iterations=1000):
        self.learning_rate=learning_rate
        self.iterations=iterations
        self.c=None
        self.m=None

    def fit(self,X,y):
        samp,features=X.shape
        self.c=np.zeros(features)
        self.c=0

        for i in range(self.iterations):
            y_pred=self.c+np.dot(X,self.m)
            d_m=(1/samp)*np.dot(X.T,(y_pred-y))
            d_c=(1/samp)*np.sum(y_pred-y)

            self.c=self.c-self.learning_rate*d_c
            self.m=self.m-self.learning_rate*d_m
    
    def predict(self,X):
        y_pred1=self.c+np.dot(X,self.m)
        return y_pred1