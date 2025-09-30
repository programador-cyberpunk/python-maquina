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

