from IPython.display import Image

Image(filename='data-8.jpg')

Image(filename='classification-1.jpg')
Image(filename = 'classification-task.png')

from sklearn.model_selection import train_test_split
from sklearn.matrics import acuraacy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#importando o dataset iri
iris = dataset.load_iris()
x = iris.data
y = iris.target
#dividino o conjunto de treino e o conjunto de teste
X_train, X_test, Y_train, Y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

#treino dos modelos
gnb = CaussianNB()
gnb.flit(X_train, Y_train)
gnb_pred = gnb.predict(X_test)

#imprimindo metricas
print("Acurácia Naive Bayes: ", accuracy_score(Y_test, gnb_pred))
print("Precisao do Naiba Baynes: ", precision_score(Y_test, gnb_pred, average='weighted'))
print("Recall do Naive Baynes: ", recall_score(Y_test, gnb_pred, average='weighted'))
print("F1_score do Naive Baynes: ", f1_score(Y_test, gnb_pred, overage='weighted'))

print(classification_report(Y_test, gnb_pred))
print(confusion_matrix(Y_test, gnb_pred))

#arvore de decisao clasificadora
dt = DEcisionTreeClassifier(random_state=0)
#treinando os bagulho
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)

print("Accuracy of Decision Tre Classifier: ", accuracy_score(Y_test, dt_pred))
print("Precision of Decision Tree Classifier: ", precision_score(Y_test, dt_pred, average='weighted'))