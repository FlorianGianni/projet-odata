import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

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

df = [df1, df2, df3, df4, df5]

####  2.Etude prealable : Comparaison des methodes de clustering sur des donnees simulees

### 2.3 Experimentation

### Affichage des nuages de points

def afficher(df, i, c=None, colormap=None):
    df.plot.scatter(x=0, y=1, c=c, colormap=colormap)
    plt.savefig('./results/etude_prealable/raw-' + str(i) + '.png')

afficher(df1, 1)
afficher(df2, 2)
afficher(df3, 3, c=2, colormap='plasma')
afficher(df4, 4, c=2, colormap='plasma')
afficher(df5, 5, c=2, colormap='plasma')

plt.close(fig='all')


### K-means

def kmeans(df, i, n_clusters):
    X = df.to_numpy()
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    plt.savefig('./results/etude_prealable/kmeans-' + str(i) + '.png')


kmeans(df1, 1, n_clusters=2)
kmeans(df2, 2, n_clusters=1)
kmeans(df3, 3, n_clusters=2)
kmeans(df4, 4, n_clusters=7)
kmeans(df5, 5, n_clusters=3)

plt.close(fig='all')


### CHA

def cha(df, i, method, metric, t):
    X = df.to_numpy()
    Z = StandardScaler().fit_transform(X)
    Z1 = hierarchy.linkage(Z, method=method, metric=metric)
    # dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
    # plt.savefig('./results/etude_prealable/cha-dendrogram-' + str(i) + '.png')
    cluster = hierarchy.fcluster(Z1, t=t, criterion='distance')
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    plt.savefig('./results/etude_prealable/cha-' + str(i) + '.png')


cha(df1, 1, method='ward', metric='euclidean', t=20)
cha(df2, 2, method='ward', metric='euclidean', t=60)
cha(df3, 3, method='ward', metric='euclidean', t=25)
cha(df4, 4, method='ward', metric='euclidean', t=9)
cha(df5, 5, method='ward', metric='euclidean', t=20)

plt.close(fig='all')


### Methode Gausienne

def gaussian(df, i, n_components):
    X = df.to_numpy()
    cluster = GaussianMixture(n_components=n_components, random_state=0).fit_predict(X)
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    plt.savefig('./results/etude_prealable/gaussian-' + str(i) + '.png')


gaussian(df1, 1, n_components=2)
gaussian(df2, 2, n_components=1)
gaussian(df3, 3, n_components=2)
gaussian(df4, 4, n_components=7)
gaussian(df5, 5, n_components=3)

plt.close(fig='all')


### DBScan

def dbscan(df, i, eps, min_samples):
    X = df.to_numpy()
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    plt.savefig('./results/etude_prealable/dbscan-' + str(i) + '.png')

dbscan(df1, 1, eps=10, min_samples=4)
dbscan(df2, 2, eps=25, min_samples=4)
dbscan(df3, 3, eps=2.7, min_samples=3)
dbscan(df4, 4, eps=2, min_samples=4)
dbscan(df5, 5, eps=2, min_samples=4)

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
