#importando os bagulho
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model import naive_bayes



#deiniçoes
wine = load_wine()
x = wine.data
y = wine.target
df = pd.DataFrame(data=X, columns=wine.feature_names)
df['target'] = Y
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,digits.data,digits.target,cv=10, scoring='accuracy')
print(scores.mean())

#stendo e treinando os bagulho
k_range = list(range(1,25))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,digits.data,digits.target,cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

data = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
kfold = KFold(5, schuffle = True)
for train, test in kfold.split(data):
    print("Train: %s", train, "Test: %s",%(data[train], data[test]))