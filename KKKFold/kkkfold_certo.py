#arrumando as tralha

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.models import metrics


#sets e treinos
wine = load_wine()
X = wine.data
Y = wine.target
X_treino, X_teste,Y_treino, Y_teste = train_test_split(X,Y, test_size=0.5,random_state=42, stratify= Y)
knn_holdout = KNeighborsClassifier(n_neighbors=5)
nb_holdout = GaussianNB()
df = pd.DataFrame(data=X, columns=wine.feature_names)
df['target'] = Y
print("\nPrimeiras linhas do bagulho: ")
print(df.head())
print(f"total das amostras: {X.shape[0]} | Total das caracteristicas: {X.shape[1]}")
print(f"\nTamanho do conjunto de Treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de Teste: {X_test.shape[0]} amostras")


print("Resultados hold-out (treino e teste")
print(f"acuracia KNN (k=5) no teste: {knn_score_holdout:.4f}")
print(f"acuracia KNN (k=5) no teste: {nb_score_holdout:.4f}")
print(f"A acuracia desse role de naive baiyes eh: {accuracy_nb:.4f}")
print(" validaçao do modelo K-Fold agora (k=10): ")

#as avaliaçoes e metodos
kfold_cv = Kfold(n_splits=10, schuffle=True,random_state=42)
n_splits = 10
knn_cv = KNeighborsClassifier(n_neighbors=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_treino,Y_treino)
nb_model = GaussianNB()
nb_model_fit(X_treino, Y_treino)
nb_cv = GaussianNB()
accuracy_nb = accuracy_score(Y_teste, Y_pred_nb)
knn_scores = cross_val_score(knn_cv,X,Y, cv=kfold_cv, scoring='accuracy')
knn_scores_cv = cross_val_score(KNeighborsClassifier(n_neighbors=5), X,Y, cv=kfold_cv,scoring='accuracy')

#medindo a acuracia desse bagulho

print("\n KNN (k=5) - pontos K-Fold: ")
for i, score in enumarete(knn_scores_cv):
    print(f" Fold {i+1}: "{score:.4f})
nb_scores = cross_val_score(nb_cv, X,Y,cv=kfold_cv,scoring='accuracy')
mean_knn_cv = knn_score_cv.mean()
print(f"a meiuca da acuracia KNN (K-Fold): {mean_knn_cv:.4f} ( +/. {knn_score_cv.std()*2:.4f})")
nb_scores_cv = cross_val_score(GaussianNB(), X,Y, cv=kfold, scoring='accuracy')
print("\n Agora vbem o Naive Bayes: ")
for i, score in enumarete(nb_scores_cv):
    print(f" Fold {i+1}: {score:.4f}")
mean_nb_cv = nb_scores_cv.mean()
print(f"Media dessa porra de acuracia do Naive Bayes: {mean_nb_cv:.4f} (+/. {nb_scores_cv.std()*2:.4f})")


#previsao
Y_pred_knn = Knn_model.predict(X_teste)
Y_pred_nb = nb_model_predict(X_teste)

k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X,Y, cv=kfold_cv,scoring='accuracy')
    k_scores.append(scores.mean())
 #exibindo todo o role   
plt.figure(figsize=(10,6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('Valor de K para KNN')
plt.ylabel('Acurácia Média Validada Cruzada')
plt.title('Otimização do K para o KNN')
plt.grid(True)
plt.show()