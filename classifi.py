from IPython.display import Image
Image(filename='classification-1.png')
Image(filename='classification-task.png')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
gnb_pred = gnb.predict(X_test)

print("Acuracia do Naive Bayes: ",accuracy_score(Y_test, gnb_pred))
print("Precisao do Naive Bayes: ",precision_score(Y_test, gnb_pred, average='weighted'))
print("Recall do Naive Bayes: ",recall_score(Y_test, gnb_pred, average='weighted'))
print("F1-Score do naive Bayes: ",f1_score(Y_test, gnb_pred, average='weighted'))
print(classification_report(Y_test, gnb_pred))
print(confusion_matrix(Y_test, gnb_pred))

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)

print("Acuracia da Arvore de Decisao: ",accuracy_score(Y_test, dt_pred))
print("Precisao da Arvore de Decisao: ",precision_score(Y_test, dt_pred, average='weighted'))
print("Recall da Arvore de Decisao: ",recall_score(Y_test, dt_pred, average='weighted'))
print("F1-Score da Arvore de Decisao: ",f1_score(Y_test, dt_pred, average='weighted'))
print(classification_report(Y_test, dt_pred ))
print(confusion_matrix(Y_test, dt_pred))

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train, Y_train)
svm_clf_pred = svm_clf.predict(X_test)

print("Acuracia do SVM: ",accuracy_score(Y_test, svm_clf_pred))
print("Precisao do SVM: ",precision_score(Y_test, svm_clf_pred, average='weighted'))
print("Recall do SVM: ",recall_score(Y_test, svm_clf_pred, average='weighted'))
print("F1-Score do SVM: ",f1_score(Y_test, svm_clf_pred, average='weighted'))
print(classification_report(Y_test, svm_clf_pred))
print(confusion_matrix(Y_test, svm_clf_pred))