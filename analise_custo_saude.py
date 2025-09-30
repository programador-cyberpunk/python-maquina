import pandas as pd

#importações e verificações padrao

df = pd.read_csv("insurance.csv")

df.head(), df.tail()

df.info()
df.describe()

#verificar dados ausentes
df.isnull().sum()

#visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns

#as caxinha pra cada item a ser analisado
plt.figure(figsize=(8,6))
sns.histplot(df['charges'],bins=30, kde=True)
plt.title('Distribuiçao dos custos do seguro')
plt.xlabel('charges')
plt.ylabel('frequencia')
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='age', y='charges')
plt.title('Relacao entre bmi e custos do seguro (charges)')
plt.xlabel('BMI')
plt.ylabel('charges')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='bmi', y='charges')
plt.title('Relacao entre bmi e custos do seguro')
plt.xlabel('age')
plt.ylabel('charges')
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(x='sex', y='charges', data=df)
plt.title('Distribuiçao de charges por sexo')
plt.xlabel('sex')
plt.ylabel('charges')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='smoker', y='charges', data=df)
plt.title('Distribuiçao de charges por fumante (smoker)')
plt.xlabel('Smoker')
plt.ylabel('charges')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='region', y='charges', data=df)
plt.title('Distribuiçao de charges por regiao')
plt.xlabel('region')
plt.ylabel('charges')
plt.show()

#pre processamento e tratamento de dados das paradas
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, colmuns['region'], drop_first=True)

#analisando a correlaçao
corr_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, camp='coolwarm', fmt= '.2f',linewidth=0.5)
plt.title('Matriz de correlacao')
plt.show()

#hora de separar as variaves
X = df.drop('charges', axis=1)
Y = df['charges']

#chegaram as porra dos teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#escalonando os dados

from sklearn.preprocessing import StandardScaler

#vamos isolar as colunas
numeric_features = ['age', 'bmi', 'children']
X_train_num = X_train[numeric_features]
X_teste_num = X_teste[numeric_features]

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_teste_num_scaled = scaler.transform(X_teste_num)

#construindo e avaliando a porra da regressão
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#avaliaçao de performance

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

#amostrando as metrica
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2: {r2:.2f}')