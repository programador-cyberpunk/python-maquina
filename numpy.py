import numpy as np
minha_lista = [1,2,3]
minha_lista

np.array(minha_lista)
matriz=[[1,2,3],[4,5,6],[7,8,9]]
matriz

np.array(matriz)
np.array

np.range(0,10)
np.range(0,10,2)

np.zeros(3)
array=np.zeros(3)
array

array=np.zeros((5,5))
array

np.ones((3,3))
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])

np.eye(4)
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])

np.linspace(0,10,2)
array([0.,10.])

np.linspace(0,10,3)
array([0.,4.,10.])

np.linspace(0,10,5)
array([0. , 2.5, 5. , 7.5,10.])

np.linspace(2.0,3.0 num=5,retstep=True)
(array([2.  ,2.25, 2.5 ,2.75, 3.  ]), 0.25)

np.random.rand(5)
array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ])