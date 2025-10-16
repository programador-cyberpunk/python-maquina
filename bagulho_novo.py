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

