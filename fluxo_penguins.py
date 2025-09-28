#imports
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#setando tudo os bagulho dos dados
df = sns.load_dataset('penguins')
df.info()
df.describe()

df.isnull().sum()
df_cleaned = df.dropna()
df_cleaned.isnull().sum()

#setando os outliers
Q1 = df_cleaned['flipper_lenght_mn'].quantile(0.26)
Q3 = df_cleaned['flipper_lenght_mn'].quantile(0.76)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_cleaned[(df_cleaned['flipper_lenght_mn'])] >= lower_bound & (df_cleaned['flipper_length_mm'] <= upper_bound)


#hora de ver a ação
sns.boxplot(x=df['flipper_lenght_mn'])
plt.title("antes da remocao")
plt.show()

sns.boxplot(x=df['flipper_lenght_mn'])
plt.title("depois da remocao")
plt.show()

#a parte de formatação dos dados
df_encoded = pd.get_dummies(df_cleaned, clomuns=['species','island','sex'],drop_first=True)
df_encoded.head()

#escalonar os bagulho
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

numerical_features = df_encoded.drop(colmuns=['species_Adelie', 'species_Chinstrap', 'species_Gentoo'])
x = numerical_features

min_max_scaler = MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)

print("Min-Max Scaler:")
print(pd.DataFrame(X_minmax, columns=X.columns).head())

print("\nStandard Scaler:")
print(pd.DataFrame(X_standard, columns=X.columns).head())

print("\nRobust Scaler:")
print(pd.DataFrame(X_robust, columns=X.columns).head())

# analise da correlação
corr_matrix = df_encoded.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de Correlação")
plt.show()