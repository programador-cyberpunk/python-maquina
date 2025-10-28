import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
try:
    df = pd.read_csv(url, sep=";")
    
    print("dados carregados nessa porra")
    
    print("Primeiras linhas: ")
    print(df.head())
    
    print("\nInformações do DataFrame: ")
    df.info()
except Exception as e:
    print(f"deu bosta ao carregar os arquivos: {e}")
    print("Verifica essas porra de novo")
    
    