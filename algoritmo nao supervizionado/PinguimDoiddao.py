# impotando os bagulho
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#exibition
df = sns.load_dataset('penguins')
df.info()
df.descbribe()
colunas = ['bill_lenght_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

#treinando e dividindo
X_treino,X_teste,y_treino,y_teste = train_test_split(df[ colunas],df['species'],test_size=0.3,random_state=42)
#primeiro o KFold
kf = KFold(n_splits=5,shuffle=True,random_state=42)
lr = LogisticRegression(max_iter=200)
lr.fit(X_treino,y_treino)

rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_treino,y_treino)
scores = cross_val_score(RandomForestClassifier(n_estimators=5),df[colunas],df['species'],cv=10)
print("Validação dos valores cuzados ai: ",np.average(scores))