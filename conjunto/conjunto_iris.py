from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#setando  e avaliando os bagulho
x, y = load_iris(return_x_y=true)
x_train, x_teste, y_train, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
single_tree = DecisionTreeClassifier(max_depth=3, max_features=3)
single_tree.fit(x_train, y_train)
single_tree_predictions = single_tree.predict(x_teste)
single_tree_accuracy = accuracy_score(y_teste, single_tree_predictions)

#treino iniciado
simple_bag = simple_bag(base_estimator=DecisionTreeClassifier(max_depth=3, max_features='sqrt'), n_estimators=50, subset_size=0.8)
simple_bag.fit(x_train, y_train)
simple_bag_predictions = simple_bag.predict(x_teste)
simple_bag_accuracy = accuracy_score(y_teste, simple_bag_predictions)

print(f'Precisao do modelo de arvre unica: {single_tree_accuracy:.2f}')
print(f'Precisao do modelo Simplebag: {simple_bag_accuracy:.2f}')