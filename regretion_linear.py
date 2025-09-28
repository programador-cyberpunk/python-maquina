import matiplotlib.pyplot as plt
import numpy as np
import pandas as pd

#os valores de exemplo x que vao de 0 a 19
x = np.range(0, 19)
#matriz
A = np.aray([x, np.ones(10)])
y = [22,24,23,20,23,30,22,24,24,20,21,30,24,20,28,20,22,24,27]
x
print(A)

#sei la oque isso faz
w=np.linalg.çstsq(A.T, y)[0]
print(w)

linha = w[0]*x+w[1]
plt.plot(x, linha, 'm-')
plt.plot(x,y, '8')
plt.show()
p19 = w[0]*19+w[1]
p19

#analise estatistica do quadro de dados (nao entendi nada)
import seaborn as sns

casas = pd.read_csv('USA_Housing.csv')
casas.info()
casas.head()
casas.tail()
casas.describe().transpose()

#analise bivariada
sns.boxplot(casas)

Q1 = casas.quantile(0.25, numeric_only=True)
Q3 = casas.quantile(0.75, numeric_only=True)
IRQ = Q3-Q1
print(Q1)
print(Q3)
print(IRQ)

contador = casas[(casas<(Q1 - 1.5*IRQ)) | (casas > (Q3+1.5*IRQ))].count()
df_contagem = pd.DataFrame(contador, columns=['Contagemd e outliers'])
df_contagem

sns.boxplot(casas, showfliers = False)
sns.pairplot(casas)
sns.heatmap(casas.corr(numeric_only=True))

casas. head(2)

x = casas[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = casas['Price']

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.7, random_state=50)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_treino, y_treino)

LinearRegression()
predicao = LR.predict(x_teste)
plt.scatter(y_teste, predicao)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_true=y_teste, y_pred = predicao)
mse = mean_squared_error(y_true=y_teste, y_pred = predicao)
rmse = mean_squared_error(y_true=y_teste, y_pred = predicao, squared=False)

print("mae: " ,mae)
print("mse: ", mae)
print("rmse: ", rmse
      )