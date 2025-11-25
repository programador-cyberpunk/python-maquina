#arrumando as tralha

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#sets e treinos
wine = load_wine()
X = wine.data
Y = wine.target
X_treino, X_teste,Y_treino, Y_teste = train_test_split(X,Y, test_size=0.5,random_state=42)
knn_holdout = KNeighborsClassifier(n_neighbors=5)
nb_holdout = GaussianNB()

print("Resultados hold-out (treino e teste")
print(f"acuracia KNN (k=5) no teste: {knn_score_holdout:.4f}")
print(f"acuracia KNN (k=5) no teste: {nb_score_holdout:.4f}")

#agora definindo e instanciando o kfold
kfold_cv = Kfold(n_splits=10, schuffle=True,random_state=42)
knn_cv = KNeighborsClassifier(n_neighbors=5)
nb_cv = GaussianNB()
knn_scores = cross_val_score(knn_cv,X,Y, cv=kfold_cv, scoring='accuracy')
nb_scores = cross_val_score(nb_cv, X,Y,cv=kfold_cv,scoring='accuracy')

#otimizando o K para KNN
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