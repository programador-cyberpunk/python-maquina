Image(ilename='img_2.png')

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class SimpleMultiClassBoosting(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=50):
        """inicializa a classe dos bagulho com o parametro fonrecido:
        parametro: base_estimator (objeto): O estimador base a ser usado. Se não for␣
↪fornecido, usa DecisionTreeClassifier.
n_estimators (int): O número de estimadores base no conjunto."""
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier(max_depth=1)
        self.n_estimator = n_estimators
        self.learners = []
        self.learners_weights = []
        self.label_encoder = LabelEncoder()
        
    def fit(self, x, y):
        """Treinando os leaners do conjunto pra funcionar as parada
        paametro: X (array-like): Conjunto de dados de treinamento.
y (array-like): Rótulos de treinamento."""

# Converte os rótulos para [0, n_classes-1]
y_encoded = self.labal_encoder.fit_transform(y)
n_classes = len(self.label_encoder.classes_)

# agora inicia essa porra
sample_weights = np.full(x.shape[0], 1 / x.shape[0])

for _ in range(self.n_estimators):
    learner = clone(self.base_estimator)
    learner.fit(x, y_encoded, sample_weight=sample_weights)
    learner.pred = learner.predict(x)
    
    #esse bagulinho aqui calcula a taxa de erro ponderada
    incorrect = (learner.pred != y_encoded)
    learner_error = np.mean((np.avarage(incorrect, weights=sample_weights)))
    learner_weight = np.log((1 - learner_error) / (learner_error + 1e-10)) + np.log(n_classes - 1) if learner_error >= 1 -(1/ n_classes):
        break
    #aumenta os pesos classiicados como incorretos
    sample.weights *= np.exp(learner_weight * incorrect > * (sample_weights 0))
      sample_weights /=np.sum(sample_weights) #normaliza os pesos
      
      #ai salve os bang
      self.learners.append(learner)
      self.learners_weights.append(learner_weight)
de predict(self, x): 
                        """
                        Gera previsões para novos dados usando os learners treinados.
                        Parâmetros:
                        X (array-like): Conjunto de dados de teste.
                        Retorna:
                        array: Previsões finais do conjunto.
                        """
      #vscode viado do caralho, mas calcula as previsoes aqui
    learner_preds = np.array([learner.predict(x) for learner in self.learners])
    weighted_preds = np.zeros((x.shape[0], len(self.label_encoder.classes_)))
    for in range(len(self.learners)):
        weighted_preds[np.arange(x.shape[0]), learner_preds[i]] += self.learners_weights[i]
        y_pred = np.argmax(weighted_preds, axis=1)
        return self.label_encoder.inverse_transform(y_pred)