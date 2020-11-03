import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

### Importation des données
df1 = pd.read_csv('./data/g2-2-20.txt', sep='     ', header=None, engine='python')
df2 = pd.read_csv('./data/g2-2-100.txt', sep='     ', header=None, engine='python')
df3 = pd.read_csv('./data/jain.txt', sep='\t', header=None)
df4 = pd.read_csv('./data/Aggregation.txt', sep='\t', header=None)
df5 = pd.read_csv('./data/pathbased.txt', sep='\t', header=None)


### Affichage des nuages de points
df1.plot.scatter(x=0, y=1)
plt.show()

df2.plot.scatter(x=0, y=1)
plt.show()

df3.plot.scatter(x=0, y=1, c=2, colormap='plasma')
plt.show()

df4.plot.scatter(x=0, y=1, c=2, colormap='plasma')
plt.show()

df5.plot.scatter(x=0, y=1, c=2, colormap='plasma')
plt.show()


### K-means
X = df1.to_numpy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
cluster = kmeans.predict(X)
df1.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.show()

X = df2.to_numpy()
kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
cluster = kmeans.predict(X)
df2.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.show()

X = df3.to_numpy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
cluster = kmeans.predict(X)
df3.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.show()

X = df4.to_numpy()
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
cluster = kmeans.predict(X)
df4.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.show()

X = df5.to_numpy()
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
cluster = kmeans.predict(X)
df5.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.show()
