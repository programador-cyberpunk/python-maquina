#importando os bang
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import kmeans_plusplus
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score


#preenchendo
Caracteristicas, Labels = make_blobs(
    n_samples = 300,
    centers=4,
    cluster_std=1.0,
    random_state=56
)

#setando as figurinha e a base de dados
plt.figure(figsize=(10,6))
plt.scatter(Caracteristicas[:,0], c=Labels)# c é de colors caraio
scalar = StandardScaler()
scalar_caracteristicas = scalar.fit_transform(Caracteristicas)
scalar_caracteristicas[:10]
Caracteristicas[:10]

kmeans=KMeans(
    init="random",
    n_clusters=4,
    n_init=20,
    max_iter=800,
    random_state=56
)
kmeans.fit(scalar_caracteristicas)
kmeans.intertia_
kmeans.cluster_centers_
kmeans.n_iter_
kmeans.labels_[:10]
Labels[:10]

#arrumando cores e graficos pra ver melhor
f, (eixo1, eixo2) = plt.subplots(1,2, sharey=True, figsize=(10,6))
eixo1.set_title('K Means')
eixo1.scatter(scalar_caracteristicas[:,0],scalar_caracteristicas[:,1],c=Labels)
eixo1.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], 
            s=30, c='red', label='Centroides' )
eixo2.set_title('Inicia')
eixo2.scatter(Caracteristicas[:,0], Caracteristicas[:,1], c=Labels)
kmeans.fit(Caracteristicas)
f,(eixo1, eixo2)=plt.subplots(1, 2, sharey=True, figsize=(10,6))
eixo1.set_title('K medias')
eixo1.scatter(Caracteristicas[:,0],Caracteristicas[:,1],c=Labels)
eixo1.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
s=50, c='red',label='Centroides')
classification_report,
eixo2.set_title('Inicia')
eixo2.scatter(Caracteristicas[:,0],Caracteristicas[:,1],c=Labels)

#exibition dos bang
print('Matriz de confusão')
print(confusion_matrix(Labels, kmeans.labels_))
kmeans_valores={
    'init' : 'random',
    'n_init':10,
    'max_iter':300,
    'random_state':56
}
SER=[] #que porra é SER????
for k in range(1,11):
    kmeansCT=KMeans(n_clusters=k, **kmeans_valores)
    kmeansCT.fit(scalar_caracteristicas)
    SER.append(kmeansCT.intertia_)
    plt.plot(range(1,11),SER)

    coeficiente_silueta=[]
    for k in range(2,11):
        kmeans_S=KMeans(n_clusters=k, **kmeans_valores)
        kmeans_S.fit(scalar_caracteristicas)
        score=silhouette_score(scalar_caracteristicas,kmeans_S.labels_)
        coeficiente_silueta.append(score)

  #agora mostra essa merda
plt.plot(range(2,11), coeficiente_silueta)
clustering = AgglomerativeClustering().fit(Caracteristicas)

# Aplicando o algoritmo de cluster aglomerativo
agg_clustering = AgglomerativeClustering(n_clusters=4)
agg_labels = agg_clustering.fit_predict(scalar_caracteristicas)

# Visualizando os clusters formados

plt.figure(figsize=(10, 6))
plt.scatter(scalar_caracteristicas[:, 0], scalar_caracteristicas[:, 1], c=agg_labels)
plt.title('Clusters Formados pelo Algoritmo Aglomerativo')
plt.show()
print('Matriz de confusao')
print(confusion_matrix(Labels, agg_labels))

print('Relatorio de classificaçao')
print(classification_report(Labels, agg_labels))

linked = linkage(scalar_caracteristicas, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked,
orientation='top',
distance_sort='descending',
show_leaf_counts=True)
plt.title('Dendograma')
plt.show()

#criando o bagulho
dbscan = DBSCAN(eps=0.3, min_samples=8)
dbscan_labels = dbscan.fit_predict(scalar_caracteristicas)
# Visualizando os clusters formados
plt.figure(figsize=(10, 6))
plt.scatter(scalar_caracteristicas[:, 0], scalar_caracteristicas[:, 1],c=dbscan_labels)
plt.title('Clusters Formados pelo Algoritmo DBSCAN')
plt.show()

print('Matriz de confusao')
print(confusion_matrix(Labels, dbscan_labels))

sc = silhouette_score(scalar_caracteristicas,dbscan_labels)
print("Silueta coeficiente: %0.2f" %sc)
ari = adjusted_rand_score(Labels, dbscan_labels)
print("Ajuste do Index Randomico: %0.2f" %ari)