# importar os bang
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_atasets import load_wine
from sklearn.rpeprocessing import StandarScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GassuianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings

#vou carregar e setar as parada
warnings.filterwarnings("ignore", category=FutureWarning)
resultados = {}

print("1 Carregando essas porra")
wine = load_wine()
X = wine.data
Y = wine.target

df = pd.DataFrame(data=X, columns=wine.feature_names)
df['target'] = Y

print("Primeiras linhas: ")
print(d.head())

print("\n Balanceano esses bagulho: ")
print(df['target'].value_counts().sort_index())
print("-" * 50)

#agora vou dividir as porra dos dados
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
print("--- Padronizando os dados ---")
scaler = StandarScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Dados paronizaos,tudo suave aqui")
print("-" * 50)

#os moelos originais agora
print("3 Treinando os modelos originais")

knn_base = KNeighborsClassifier(n_neighbors=5)
knn_base.fit(X_train_scaled, Y_train)
Y_pred_knn_base = knn_base.predict(X_test_scaled)
acc_knn_base = accuracy_score(Y_test, Y_pred_knn_base)
resultados['KNN (base)'] = acc_knn_base
print("Relatorio KNN (base): ")
print(classification_report(Y_test, Y_pred_knn_base, target_names=wine.target_names))

nb_base= GassuianNB()
nb_base.fit(X_train_scaled, Y_train)
Y_pred_nb_base = nb_base.predict(X_test_scaled)
acc_nb_base = accuracy_score(Y_test, Y_pred_nb_base)
resultados['Naive Bayes (base)'] = acc_nb_base

print("Relatorio Naive Bayes (base): ")
print(classification_report(Y_test, Y_pred_nb_base, target_names=wine.target_names))
print("-" * 50)

#hora da sobreamostragem com SMOTE
print("4 Treinando os modelos com SMOTE")
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train_scaled, Y_train)
print("Balanceamento  das classes depois o SMOTE: ")
print(pd.Series(Y_train_smote).value_counts().sort_index())

knn_smote = KNeighborsClassifier(n_neighbors=5)
knn_smote.fit(X_train_smote, Y_train_smote)
Y_pred_knn_smote = knn_smote.predict(X_test_scaled)
acc_knn_smote = accuracy_score(Y_test, Y_pred_knn_smote)
resultados['KNN (SMOTE)'] = acc_knn_smote

print("Relatorio KNN (SMOTE): ")
print(classification_report(Y_test, Y_pred_knn_smote, target_names=wine.target_names))

nb_smote = GassuianNB()
nb_smote.fit(X_train_smote, Y_train_smote)
Y_pred_nb_smote = nb_smote.predict(X_test_scaled)
acc_nb_base = accuracy_score(Y_test, Y_pred_nb_smote)
resultados['Naive Bayes (SMOTE)'] = acc_nb_smote

print("Relatorio Naive Bayes (SMOTE): ")
print(classification_report(Y_test, Y_pred_nb_smote, target_names=wine.target_names))
print("-" * 50)

#aplicar a subamostragem randomica
print("5 Treinando os modelos com Random Under Sampler" )
rus = RandomUnderSampler(random_state=42)
X_train_rus, Y_train_rus = rus.fit_resample(X_train_scaled, Y_train)

print("Balanceamento das classes depois do RUS: ")
print(pd.Series(Y_train_rus).value_counts().sort_index())

knn_rus = KNeighborsClassifier(n_neighbors=5)
knn_rus.fit(X_train_rus, Y_train_rus)
Y_pred_knn_rus = knn_rus.predict(X_test_scaled)
acc_knn_rus = accuracy_score(Y_test, Y_pred_knn_rus)
resultados['KNN (RUS)'] = acc_knn_rus

print("Relatorio KNN (RUS): ")
print(classification_report(Y_test, Y_pred_knn_rus, target_names=wine.target_names))

nb_rus = GassuianNB()
nb_rus.fit(X_train_rus, Y_train_rus)
Y_pred_nb_rus = nb_rus.predict(X_test_scaled)
acc_nb_rus = accuracy_score(Y_test, Y_pred_nb_rus)
resultados['Naive Bayes (RUS)'] = acc_nb_rus

print("Relatorio Naive Bayes (RUS): ")
print(classification_report(Y_test, Y_pred_nb_rus, target_names=wine.target_names))
print("-" * 50)

# finalmente,analisar e mostrar os resultados,cabou essa porra
print("6 Resultados Finais: ")
print("Acuracias dos modelos: ")

for modelo, acc in resultados.items():
    print(f"{modelo}: {acc:.4f}")

print("-" * 50)    