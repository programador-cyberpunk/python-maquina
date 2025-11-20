from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#conjuntos carregados
x, y = load_iris(return_x_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, max_features='sqrt'), n_estimators=50, max_samples=0.8, random_state=42)
bagging_model.fit(x_train, y_train)
predictions = bagging_model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

#imprimindo a precisao do Bagging
print(f'Precisao de modelo Bagging: {accuracy:.2f}')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
