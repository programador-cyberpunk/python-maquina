from IPython.display import Image
Image(filename='2.png')
Image(filename='3.png')
Image(filename='4.png')
Image(filename='5.png')

#construindo o exemplo e importando os bagulho
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#carregation e dividation
digits = load_digits()


x_treino, x_teste,y_treino, y_teste = train_test_split(digits.data, digits.targer, test_size=0.3)

#usando dierentes algoritmos agora

#regressao logistica
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(x_treino, x_teste)
print("Score: ",lr.score(x_teste,y_teste))

#svc
svm = SVC(gamma='auto')
svm.it(x_treino, y_treino)
print("Score: ",svm.score(x_teste, y_teste))

#random forest
rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_treino, y_treino)
print("Score: ",rf.score(x_teste, y_teste))

#cross_val
score_lr = cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), digits.data, digits.target, cv=5)
print(score_lr)
print("Avg: " , np.mean(score_lr))

#definicao de svm e cv
score_svm = cross_val_score(SVC(gamma='auto'), digits.data, digits.target, cv=5)
print(score_svm)
print("Avg: " , np.mean(score_svm))

#com random orest 
score_rf = cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=5)
print(score_rf)
print("Avg: " , np.mean(score_rf))