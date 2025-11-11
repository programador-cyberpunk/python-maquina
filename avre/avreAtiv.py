import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt

# definino tudo certinho
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# dadinho ze pequeno carregado
df = sns.load_dataset('penguins')
df.head()
df_limpo = df.dropna()

features['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
target = 'species'
x = df_limpo[features]
y = df_limpo[target]

#treininhos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f'Total de dados: {len(df_limpo)}')
print(f'Dados de treino: {len(x_train)}')
print(f'Dados de teste: {len(x_test)}')

# vamo começar as parada
avre = DecisionTreeClassifier(random_State=42)
avre.fit(x_train, x_test)
avre_mae_dina = avre.predict(x_test)#ve o futuro #previsoes
avre_mira = accuracy_score(y_test, avre_mae_dina)#precisao

#vamo mostra os bagulho tudo bonitinho
print(f'Precisao de arvore de decisao: {avre_mira * 100:2f%}')

#modelo floresta
floresta = RandomForestClassifier(n_estimators=100,random_state=42)

#treino
floresta.fit(x_train,x_test)
floresta_mae_dina = floresta.predict(x_test)#previsoes
floresta_mira = accuracy_score('floresta_mae_dina * 100:.2f')#precisao

#agora vem a hora da verdade

#primeiro a arvore
cm_avre = confusion_matrix(y_test, avre_mae_dina)
labels = avre.classes_

plt.figure(figsize=(8,6))
sns.heatmap(cm_avre, annot=True, fmt='g', cmap='Blues', xticlabels=labels
        ,yticklabels=labels)
plt.title("Matriz de confusão(pq tem esse nome?) - Arvore de Decisão")
plt.xlabel('Valor previsto')
plt.ylabel('Valor real')
plt.show()

print("----- Relatorio de classificação (Arvore) -----")
print(classification_report(y_test, avre_mae_dina))

#floresta toda
cm_floresta = confusion_matrix(y_test, floresta_mae_dina)
labels = floresta.classe

plt.figure(figsize=(8,6))
sns.heatmap(cm_floresta, annot=True,cmap='Greens', xticlabels=labels
        ,yticklabels=labels )
plt.title("Matriz de confusão- Floresta")
plt.xlabel('Valor previsto')
plt.ylabel('Valor real')
plt.show()

print("---- Rleatorio de classificação (floresta) ----")
print(classification_report(y_test,floresta_mae_dina))
