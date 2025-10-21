#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv('creditcard.csv')

#imprime os bagulho no dataframe
print(data.info())

data['normAmount'] = StandartScaler().fit_transform(np.array(data['Amount']).reshape(-1, 1))
data data.drop(['Time', 'Amount', axis = 1])
data['Class'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split((['Class'], axis=1), data['Class'], test_Size = 0.3, random_state =0)
print("Numero de transacoes do conjunto de dados X_Train: ",X_train.shape)
print("Numero de transacoes do conjunto de dados Y_Train: ",Y_train.shape)
print("Numero de transacoes do conjunto de dados X_Test: ",X_test.shape)
print("Numero de transacoes do conjunto de dados Y_test: ",Y_test.shape)


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, Y_train.ravel() predictions = lr.predic(X_test))
print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))


# In[ ]:


print("Antes do OverSampling, contagem do rotulo '1': ".format(sum(Y_train == 1)))
print("Antes do OverSampling, contagem do rotulo '0':{} \n".format(sum(Y_train == 0)))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train.ravel())
print('Depois do OverSampling, a frma do X_train: {}'.format(X_train_res.shape))
print('Depois do OverSampling, a frma do Y_train: {} \n'.format(Y_train_res.shape))
print('Depois do OverSampling, contagens do rotulo '1': {}'.format(sum(Y_train_res == 1)))
print('Depois do OverSampling, a frma do X_train: {}'.format(sum(Y_train_res.shape == 0)))


# In[ ]:


lr1 = LogisticRegression()
lr1.fit(X_train_res, Y_train_res.ravel())
predictions = lr1.predict(X_test)

print(classificatin_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))


# In[ ]:


print("antes da subamostragem, contagens do rotulo '1': {}".format(Sum(Y_train == 1)))
print("antes da subamostragem, contagens do rotulo '0': {}".format(Sum(Y_train == 0)))

from imblearn.under_sampling import NearMiss
nr = NearMiss()
X_train_miss, Y_train_miss = nr.fit_resample(X_train, Y_train())
print('Depois da subamostragem, a forma do X_train: {}'.format(X_train_miss.shape))
print('Depois da subamostragem, a forma do Y_train: {}'.format(Y_train_miss.shape))
print('Depois da subamostragem, contagens do rotulo 1: {}'.format(sum(Y_train_miss.shape == 1)))
print('Depois da subamostragem, contagens do rotulo 0: {}'.format(sum(Y_train_miss.shape == 0)))

