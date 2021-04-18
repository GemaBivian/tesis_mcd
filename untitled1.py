# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 22:04:22 2021
def cumsum(lista):
    lista2 = [lista[0]]
    for i in range(1, len(lista)):
        lista2.append(lista[i] + lista2[-1])

    return lista2
@author: gema_
"""

import numpy as np
import math
import matplotlib.pyplot as plt

N = 100
np.random.seed(133)
A = np.random.normal(0, 1, N-1)
T = 1
dt = 1/(N-1)
t = [c/(N-1) for c in range(0, int(N-1))]
dW = [c*math.sqrt(dt) for c in A]
W = np.cumsum(dW,dtype=float)

plt.plot(W)
plt.ylabel('Movimiento Browniano')
plt.xlabel('Tiempo')
plt.show()

#-----------------------
