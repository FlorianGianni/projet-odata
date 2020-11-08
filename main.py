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
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.decomposition import PCA
from statistics import mean

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
    # plt.savefig('./results/etude_prealable/raw-' + str(i) + '.png')


afficher(df1, 1)
afficher(df2, 2)
afficher(df3, 3, c=2, colormap='plasma')
afficher(df4, 4, c=2, colormap='plasma')
afficher(df5, 5, c=2, colormap='plasma')

# plt.close(fig='all')


### K-means

def kmeans(df, i, n_clusters):
    X = df.to_numpy()
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    if i>2:
        plt.title('ARI = ' + str(adjusted_rand_score(df.iloc[:,2],cluster)))
    plt.savefig('./results/etude_prealable/kmeans-' + str(i) + '.png')


kmeans(df1, 1, n_clusters=2)
kmeans(df2, 2, n_clusters=1)
kmeans(df3, 3, n_clusters=2)
kmeans(df4, 4, n_clusters=7)
kmeans(df5, 5, n_clusters=3)

def score_rand_ajuste_kmeans(df,i,n_clusters):
    X = df.to_numpy()
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    print('Score de Rand ajusté df'+ str(i), adjusted_rand_score(df[2],cluster))
    
score_rand_ajuste_kmeans(df3,3,n_clusters=2)
score_rand_ajuste_kmeans(df4,4,n_clusters=7)
score_rand_ajuste_kmeans(df5,5,n_clusters=3)
    
# plt.close(fig='all')


### CHA

def cha(df, i, method, metric, t):
    X = df.to_numpy()
    Z = StandardScaler().fit_transform(X)
    Z1 = hierarchy.linkage(Z, method=method, metric=metric)
    # dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
    # plt.savefig('./results/etude_prealable/cha-dendrogram-' + str(i) + '.png')
    cluster = hierarchy.fcluster(Z1, t=t, criterion='distance')
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    if i>2:
        plt.title('ARI = ' + str(adjusted_rand_score(df.iloc[:,2],cluster)))
    # plt.savefig('./results/etude_prealable/cha-' + str(i) + '.png')


cha(df1, 1, method='ward', metric='euclidean', t=20)
cha(df2, 2, method='ward', metric='euclidean', t=60)
cha(df3, 3, method='ward', metric='euclidean', t=25)
cha(df4, 4, method='ward', metric='euclidean', t=9)
cha(df5, 5, method='ward', metric='euclidean', t=20)

# plt.close(fig='all')

def score_rand_ajuste_cha(df,i,method,metric,t):
    X = df.to_numpy()
    Z = StandardScaler().fit_transform(X)
    Z1 = hierarchy.linkage(Z, method=method, metric=metric)
    cluster = hierarchy.fcluster(Z1, t=t, criterion='distance')
    print('Score de Rand ajusté df'+ str(i), adjusted_rand_score(df[2],cluster))

score_rand_ajuste_cha(df3,3,method='ward',metric='euclidean',t=20)
score_rand_ajuste_cha(df4,4,method='ward',metric='euclidean',t=9)
score_rand_ajuste_cha(df5,5,method='ward',metric='euclidean',t=20)


### Methode Gausienne

def gaussian(df, i, n_components):
    X = df.to_numpy()
    cluster = GaussianMixture(n_components=n_components, random_state=0).fit_predict(X)
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    if i>2:
        plt.title('ARI = ' + str(adjusted_rand_score(df.iloc[:,2],cluster)))
    # plt.savefig('./results/etude_prealable/gaussian-' + str(i) + '.png')


gaussian(df1, 1, n_components=2)
gaussian(df2, 2, n_components=1)
gaussian(df3, 3, n_components=2)
gaussian(df4, 4, n_components=7)
gaussian(df5, 5, n_components=3)

# plt.close(fig='all')

def score_rand_ajuste_gaussienne(df,i,n_components):
    X = df.to_numpy()
    cluster = GaussianMixture(n_components=n_components, random_state=0).fit_predict(X)
    print('Score de Rand ajusté df'+ str(i), adjusted_rand_score(df[2],cluster))
    
score_rand_ajuste_gaussienne(df3,3,n_components=2)
score_rand_ajuste_gaussienne(df4,4,n_components=7)
score_rand_ajuste_gaussienne(df5,5,n_components=3)

### DBScan

def dbscan(df, i, eps, min_samples):
    X = df.to_numpy()
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    df.plot.scatter(x=0, y=1, c=cluster, colormap='plasma')
    if i>2:
        plt.title('ARI = ' + str(adjusted_rand_score(df.iloc[:,2],cluster)))
    # plt.savefig('./results/etude_prealable/dbscan-' + str(i) + '.png')


dbscan(df1, 1, eps=10, min_samples=4)
dbscan(df2, 2, eps=25, min_samples=4)
dbscan(df3, 3, eps=2.7, min_samples=3)
dbscan(df4, 4, eps=2, min_samples=4)
dbscan(df5, 5, eps=2, min_samples=4)

# plt.close(fig='all')

def score_rand_ajuste_dbscan(df, i, eps, min_samples):
    X = df.to_numpy()
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    print('Score de Rand ajusté df'+ str(i), adjusted_rand_score(df[2],cluster))
    
score_rand_ajuste_dbscan(df3, 3, eps=2.7, min_samples=3)
score_rand_ajuste_dbscan(df4, 4, eps=2, min_samples=4)
score_rand_ajuste_dbscan(df5, 5, eps=2, min_samples=4)





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
colonne_nom = df['country']
df = df.drop(labels=['country'], axis=1)
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)

# Repérage des valeurs aberrantes
# colonne_gdp = df['GDP'].copy()
# colonne_gdp.sort_values(ascending = False)

print(df[df['GDP']>105000]) # valeur aberrante : PIB supérieur à 105000 valeur max PIB Luxembourg.
#Indice 7 158 159 remplacement par donnée trouvé sur internet
df['GDP'][7] = 52022
df['GDP'][158] = 39435
df['GDP'][159] = 48466

# Normalisation des variables
X = df.to_numpy()
sc = StandardScaler()
sc.fit(X)
Z = sc.transform(X)


### 3.3 Recherche de correlations

print(pd.DataFrame(Z).corr())
pd.plotting.scatter_matrix(pd.DataFrame(df))
plt.show()


### 3.4 Clustering des donnees

## K-means

# K = 12
# kmeans = KMeans(n_clusters=K, random_state=0).fit(Z)
# cluster = kmeans.predict(Z)
# df.insert(1, 'cluster', cluster)
# for i in range(K):
#     print(df[df['cluster'] == i])

def methode_coude(K):
    distorsion = []
    K = range(1,10)
    for i in K:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(Z)
        distorsion.append(kmeans.inertia_)
    plt.plot(K,distorsion,'bx-')
    plt.show()
    
def k_means_cluster_score(K):
    cluster = KMeans(n_clusters=K, random_state=0).fit_predict(Z)
    print('silhouette score kmeans K=' + str(K), silhouette_score(Z,labels=cluster))
    print('Davies bouldin score kmeans K=' + str(K), davies_bouldin_score(Z,labels=cluster))
    return (cluster)

print(k_means_cluster_score(2))
print(k_means_cluster_score(3))
print(k_means_cluster_score(4))
print(k_means_cluster_score(5))
print(k_means_cluster_score(6))

methode_coude(10)

# On choisit K=4

## CHA

def CHA_cluster_score(method, metric, t):
    Z1 = hierarchy.linkage(Z, method=method, metric=metric)
    # dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
    cluster = hierarchy.fcluster(Z1, t=t, criterion='distance')
    K = max(cluster) # Le nombre de cluster
    print('silhouette score CHA K=' + str(K), silhouette_score(Z,labels=cluster))
    print('Davies bouldin score CHA K=' + str(K), davies_bouldin_score(Z,labels=cluster))
    return(cluster)
    
print(CHA_cluster_score(method='ward', metric = 'euclidean', t=25))
print(CHA_cluster_score(method='ward', metric = 'euclidean', t=18))
print(CHA_cluster_score(method='ward', metric = 'euclidean', t=15))

# On choisit K = 4 pour t=18

def gaussienne_cluster_score(K):
    cluster = GaussianMixture(n_components=K, random_state=0).fit_predict(Z)
    print('silhouette score Gaussienne K=' + str(K), silhouette_score(Z,labels=cluster))
    print('Davies bouldin score Gausienne K=' + str(K), davies_bouldin_score(Z,labels=cluster))
    return (cluster)

print(gaussienne_cluster_score(2))
print(gaussienne_cluster_score(3))
print(gaussienne_cluster_score(4))
print(gaussienne_cluster_score(5))
print(gaussienne_cluster_score(6))

# On choisit K=4

def dbscan_cluster_score(eps, min_samples):
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    K = (max(cluster) + 1)
    print('silhouette score DBScan K=' + str(K), silhouette_score(Z,labels=cluster))
    print('Davies bouldin score DBscan K=' + str(K), davies_bouldin_score(Z,labels=cluster))
    return (cluster)

print(dbscan_cluster_score(1,4))
print(dbscan_cluster_score(2,4))
print(dbscan_cluster_score(3,4))
print(dbscan_cluster_score(4,4))
print(dbscan_cluster_score(5,4))
print(dbscan_cluster_score(6,4))

# Ici on a K=1 avec epsilon = 3 et min_points = 4


### 3.5 Clustering des données après réduction des dimensions

# Critere 1 : Methode du coude
pca_test = PCA(n_components=8) #Pour déterminer le nombre d'axe
pca_test.fit(Z)
Z_test = pca_test.transform(Z)
print(pca_test.explained_variance_)
print(pca_test.explained_variance_ratio_)

abscisse = [i for i in range(1,9)]
plt.plot(abscisse,pca_test.explained_variance_,'bx-')
plt.show()

# Critere 2 : part d'inertie à au moins 80%
print(np.cumsum(pca_test.explained_variance_ratio_)) #Part d'inertie cumulé

#Critère 3 : régle de Kaiser
def Kaiser():
    n = len(pca_test.explained_variance_ratio_)
    conserve = []
    moyenne = mean(pca_test.explained_variance_ratio_)
    for i in range(n):
        if pca_test.explained_variance_ratio_[i] > moyenne:
            conserve.append(pca_test.explained_variance_ratio_)
    return(conserve)
        
print(Kaiser())

# On converse 4 axes avec critère 2

pca = PCA(n_components=4)
pca.fit(Z)
Z_acp = pca.transform(Z)

def correlation_centre(Z,j,pca): #variable centrée zj et facteur dk à récupérer
    n = len(Z[0]) #on travaille sur les variables donc sur une colonne (donc on parcourt les lignes)
    valeurs_propres_Z, vecteurs_propres_Z = pca.explained_variance_,pca.components_ 
    correlation = []
    for i in range(n):
        cor = ((valeurs_propres_Z[j])**(1/2))*(vecteurs_propres_Z[j][i])
        correlation.append(cor)
    return(correlation)

def texte_cercle(X,Y):
    n = len(X)
    for i in range(n):
        plt.text(X[i],Y[i],df.columns[i])
    return()

def ligne_cercle(X,Y):
    n = len(X)
    for i in range(n):
        plt.arrow(0,0,X[i],Y[i])
    return()

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
cercle_1 = plt.Circle((0,0),1)
ax.add_artist(cercle_1)
texte_cercle((correlation_centre(Z,0,pca)),correlation_centre(Z,1,pca))
ligne_cercle((correlation_centre(Z,0,pca)),correlation_centre(Z,1,pca))
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
cercle_2 = plt.Circle((0,0),1)
ax.add_artist(cercle_2)
texte_cercle((correlation_centre(Z,2,pca)),correlation_centre(Z,3,pca))
ligne_cercle((correlation_centre(Z,2,pca)),correlation_centre(Z,3,pca))
plt.show()

# Interprétation des axes :

def nuage_individu(Z_acp): 
    n = len(Z_acp)
    X = np.zeros(n)
    Y = np.zeros(n)
    Z = np.zeros(n)
    T = np.zeros(n)
    for i in range(n):
        X[i] = Z_acp[i][0]
        Y[i] = Z_acp[i][1]
        Z[i] = Z_acp[i][2]
        T[i] = Z_acp[i][3]
    return(X,Y,Z,T)

def nuage_individu_texte(X,Y):
    n = len(X)
    for i in range(n):
        plt.text(X[i],Y[i],colonne_nom[i])
    return()

def k_means_acp(K):
    cluster = KMeans(n_clusters=K, random_state=0).fit_predict(Z_acp)
    print('silhouette score kmeans K=' + str(K), silhouette_score(Z_acp,labels=cluster))
    print('Davies bouldin score kmeans K=' + str(K), davies_bouldin_score(Z_acp,labels=cluster))
    plt.scatter(nuage_individu(Z_acp)[0],nuage_individu(Z_acp)[1],c=cluster,cmap='plasma')
    plt.show()
    return (cluster)

k_means_acp(4)

def CHA_acp(method, metric, t):
    Z1 = hierarchy.linkage(Z_acp, method=method, metric=metric)
    # dn = hierarchy.dendrogram(Z1) # Pour avoir le t coupure
    cluster = hierarchy.fcluster(Z1, t=t, criterion='distance')
    K = max(cluster) # Le nombre de cluster
    print('silhouette score CHA K=' + str(K), silhouette_score(Z_acp,labels=cluster))
    print('Davies bouldin score CHA K=' + str(K), davies_bouldin_score(Z_acp,labels=cluster))
    plt.scatter(nuage_individu(Z_acp)[0],nuage_individu(Z_acp)[1],c=cluster,cmap='plasma')
    plt.show()
    return(cluster)

CHA_acp(method='ward',metric='euclidean',t=18)

def gaussienne_acp(K):
    cluster = GaussianMixture(n_components=K, random_state=0).fit_predict(Z_acp)
    print('silhouette score Gaussienne K=' + str(K), silhouette_score(Z_acp,labels=cluster))
    print('Davies bouldin score Gausienne K=' + str(K), davies_bouldin_score(Z_acp,labels=cluster))
    plt.scatter(nuage_individu(Z_acp)[0],nuage_individu(Z_acp)[1],c=cluster,cmap='plasma')
    plt.show()
    return (cluster)

gaussienne_acp(4)

def dbscan_acp(eps, min_samples):
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z_acp)
    K = (max(cluster) + 1)
    print('silhouette score DBScan K=' + str(K), silhouette_score(Z_acp,labels=cluster))
    print('Davies bouldin score DBscan K=' + str(K), davies_bouldin_score(Z_acp,labels=cluster))
    plt.scatter(nuage_individu(Z_acp)[0],nuage_individu(Z_acp)[1],c=cluster,cmap='plasma')
    plt.show()
    return (cluster)

dbscan_acp(3,4)

print('Score de Rand ajuste k_means', adjusted_rand_score(k_means_cluster_score(4),k_means_acp(4)))
print('Score de Rand ajuste CHA', adjusted_rand_score(CHA_cluster_score(method='ward',metric='euclidean',t=18),CHA_acp(method='ward',metric='euclidean',t=18)))
print('Score de Rand ajuste gaussienne', adjusted_rand_score(gaussienne_cluster_score(4),gaussienne_acp(4)))
print('Score de Rand ajuste DBScan', adjusted_rand_score(dbscan_cluster_score(3,4),dbscan_acp(3,4)))
