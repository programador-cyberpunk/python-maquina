from IPython.display import Image
Image(filename='img_1.png')

from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_teste_split
from skelarn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import mode

class SimpleBag:
    def __init__(self, base_estimator = None, n_estimator=10, subset_size=0.8):
        "iniciando aqui esta porra, parametros: "
        self.base_estimator =base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1, max_features='sqrt')
        self.n_estimator = n_estimator
        self.ubset_size = subset_size
        self.base_learners = []
        self.is_fitted = False
        
    def fit(self,x,y):
        """treinnando os bagging 
        Parâmetros: X (array-like): Conjunto de dados de treinamento.
            y (array-like): Rótulos de treinamento."""
        n_samples=X.shape[0]
        subset_size = int(n_samples * self.subset_size)
        self.base_learners = []
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(range(n_samples), size=subset_size, replace=True)
            x_subset, y_subset = x[indices], y[indices]
            cloned_estimator = clone(self.base_estimator)
            cloned_estimator.fit(x_subset, y_subset)
            self.base_learners.append(cloned_estimator)
            
            self.is_fitted = True
        def predict(self, x):
            """fazendo as previsoes
            Parâmetros: X (array-like): Conjunto de dados para previsão.
            Retorna: array-like: Previsões agregadas dos estimadores base."""
            if not self.is_fitted:
                raise Exception("O modelo não foi treinado ainda. Chame o método 'fit' antes de 'predict'.")
                
            predictions = np.array([learner.predict(x) for learner in self.base_learners])
            final_predictions = mode(predictions, axis=0).mode[0]
            return final_predictions.ravel()