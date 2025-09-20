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

iris = dataset.load_iris()
x = iris.data
y = iris.target

X_train, X_test, Y_train, Y_teste = train_test_split(x, y, test_size=0.3, random_state=1)
