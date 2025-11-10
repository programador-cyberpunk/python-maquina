import numpy as np
import pandas as pd
import seaborn as sns
# definino tudo certinho
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# dadinho ze pequeno carregado
df = sns.load_dataset('penguins')
df.head()
df_limpo = df.dropna()

features['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
target = 'species'
x = df_limpo[features]
y = df_limpo[targey]

#treininhos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

