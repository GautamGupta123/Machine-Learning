import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
data=pd.read_csv("csv/Wholesale customers data.csv");

print(data.head(5));
print(data.columns)

from sklearn.preprocessing import normalize
scaleddata=normalize(data);
scaleddata=pd.DataFrame(scaleddata,columns=data.columns);
print(scaleddata.head());

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10,7))
plt.title("Dendrogram")
dend=shc.dendrogram(shc.linkage(scaleddata,method='ward'))
plt.show()

plt.figure(figsize=(10,7))
plt.title("Dendrogram");
dend=shc.dendrogram(shc.linkage(scaleddata,method='ward'));
plt.axhline(y=5,color='r',linestyle='--');
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster=AgglomerativeClustering(n_clusters=2,linkage='ward')
cluster.fit_predict(scaleddata)
plt.figure(figsize=(10,7))
plt.scatter(scaleddata['Fresh'],scaleddata['Grocery'],c=cluster.labels_)
plt.show()




