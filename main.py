import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    os.makedirs('./results/etude_prealable/')
    os.makedirs('./results/classement_pays/')
except:
    pass

### Importation des donnees
df1 = pd.read_csv('./data/g2-2-20.txt', sep='     ', header=None, engine='python')
df2 = pd.read_csv('./data/g2-2-100.txt', sep='     ', header=None, engine='python')
df3 = pd.read_csv('./data/jain.txt', sep='\t', header=None)
df4 = pd.read_csv('./data/Aggregation.txt', sep='\t', header=None)
df5 = pd.read_csv('./data/pathbased.txt', sep='\t', header=None)

####  2.Etude prealable : Comparaison des methodes de clustering sur des donnees simulees

### 2.3 Experimentation

### Affichage des nuages de points
df1.plot.scatter(x=0, y=1)
plt.savefig('./results/etude_prealable/raw-1.png')

df2.plot.scatter(x=0, y=1)
plt.savefig('./results/etude_prealable/raw-2.png')

df3.plot.scatter(x=0, y=1, c=2, colormap='plasma')
plt.savefig('./results/etude_prealable/raw-3.png')

df4.plot.scatter(x=0, y=1, c=2, colormap='plasma')
plt.savefig('./results/etude_prealable/raw-4.png')

df5.plot.scatter(x=0, y=1, c=2, colormap='plasma')
plt.savefig('./results/etude_prealable/raw-5.png')


### K-means
X = df1.to_numpy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
cluster = kmeans.predict(X)
df1.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/kmeans-1.png')

X = df2.to_numpy()
kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
cluster = kmeans.predict(X)
df2.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/kmeans-2.png')

X = df3.to_numpy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
cluster = kmeans.predict(X)
df3.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/kmeans-3.png')

X = df4.to_numpy()
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
cluster = kmeans.predict(X)
df4.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/kmeans-4.png')

X = df5.to_numpy()
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
cluster = kmeans.predict(X)
df5.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/kmeans-5.png')


### CHA
X = df1.to_numpy()
sc = StandardScaler()
Z = sc.fit_transform(X)
Z1 = hierarchy.linkage(Z,'ward')
# dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
# plt.savefig('./results/etude_prealable/cha-dendrogram-1.png')
cluster = hierarchy.fcluster(Z1,t=20,criterion='distance')
df1.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/cha-1.png')

X = df2.to_numpy()
sc = StandardScaler()
Z = sc.fit_transform(X)
Z1 = hierarchy.linkage(Z,'ward')
# dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
# plt.savefig('./results/etude_prealable/cha-dendrogram-2.png')
cluster = hierarchy.fcluster(Z1,t=60,criterion='distance')
df2.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/cha-2.png')

X = df3.to_numpy()
sc = StandardScaler()
Z = sc.fit_transform(X)
Z1 = hierarchy.linkage(Z,'ward')
# dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
# plt.savefig('./results/etude_prealable/cha-dendrogram-3.png')
cluster = hierarchy.fcluster(Z1,t=25,criterion='distance')
df3.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/cha-3.png')

X = df4.to_numpy()
sc = StandardScaler()
Z = sc.fit_transform(X)
Z1 = hierarchy.linkage(Z,'ward')
# dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
# plt.savefig('./results/etude_prealable/cha-dendrogram-4.png')
cluster = hierarchy.fcluster(Z1,t=9,criterion='distance')
df4.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/cha-4.png')

X = df5.to_numpy()
sc = StandardScaler()
Z = sc.fit_transform(X)
Z1 = hierarchy.linkage(Z,'ward')
# dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
# plt.savefig('./results/etude_prealable/cha-dendrogram-5.png')
cluster = hierarchy.fcluster(Z1,t=20,criterion='distance')
df5.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/cha-5.png')


### Methode Gausienne
X = df1.to_numpy()
Gausienne = GaussianMixture(n_components=2, random_state=0).fit(X)
cluster = Gausienne.predict(X)
df1.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/gaussian-1.png')

X = df2.to_numpy()
Gausienne = GaussianMixture(n_components=1, random_state=0).fit(X)
cluster = Gausienne.predict(X)
df2.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/gaussian-2.png')

X = df3.to_numpy()
Gausienne = GaussianMixture(n_components=2, random_state=0).fit(X)
cluster = Gausienne.predict(X)
df3.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/gaussian-3.png')

X = df4.to_numpy()
Gausienne = GaussianMixture(n_components=7, random_state=0).fit(X)
cluster = Gausienne.predict(X)
df4.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/gaussian-4.png')

X = df5.to_numpy()
Gausienne = GaussianMixture(n_components=3, random_state=0).fit(X)
cluster = Gausienne.predict(X)
df5.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
plt.savefig('./results/etude_prealable/gaussian-5.png')


### DBScan

X = df1.to_numpy()
cluster = DBSCAN(eps=10,min_samples=4).fit_predict(X)
df1.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/dbscan-1.png')

X = df2.to_numpy()
cluster = DBSCAN(eps=25,min_samples=4).fit_predict(X)
df2.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/dbscan-2.png')

X = df3.to_numpy()
cluster = DBSCAN(eps=2.7,min_samples=3).fit_predict(X)
df3.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/dbscan-3.png')

X = df4.to_numpy()
cluster = DBSCAN(eps=2,min_samples=4).fit_predict(X)
df4.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/dbscan-4.png')

X = df5.to_numpy()
cluster = DBSCAN(eps=2,min_samples=4).fit_predict(X)
df5.plot.scatter(x=0,y=1,c=cluster,colormap='plasma')
plt.savefig('./results/etude_prealable/dbscan-5.png')

plt.close(fig='all')

#### 3.Classement des principaux pays du monde en fonction de leur developpement

### 3.1 Examen des donnees

df = pd.read_csv('./data/data.csv')

print(df.info())
print(df.isna().sum())
print(df.describe())
df.hist()
plt.show()


### 3.2 Preparation des donnees

# On se debarrasse du nom des pays et on remplit les donnees manquantes avec la moyenne
X = df.drop(labels=['country'], axis=1)
X = X.apply(lambda x: x.fillna(x.mean()),axis=0)

# Normalisation des variables
X = X.to_numpy()
sc = StandardScaler()
sc.fit(X)
Z = sc.transform(X)


### 3.3 Recherche de correlations

print(pd.DataFrame(Z).corr())
pd.plotting.scatter_matrix(pd.DataFrame(df))
plt.show()


### 3.4 Clustering des donnees

## K-means

K = 12
kmeans = KMeans(n_clusters=K, random_state=0).fit(Z)
cluster = kmeans.predict(Z)
df.insert(1, 'cluster', cluster)
for i in range(K):
    print(df[df['cluster'] == i])
