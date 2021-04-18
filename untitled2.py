
'''
geometric brownian motion with drift!

Spezifikationen:

    mu=drift factor [Annahme von Risikoneutralitaet]
    sigma: volatility in %
    T: time span
    dt: lenght of steps
    S0: Stock Price in t=0
    W: Brownian Motion with Drift N[0,1] 
'''

import matplotlib.pyplot as plt
import numpy as np
T = 1
mu = 0.025
sigma = 0.1
S0 = 5
dt = 0.01
N = round(T/dt)
t = np.linspace(0, T, N)
W = np.random.standard_normal(size = N) 
W = np.cumsum(W)*np.sqrt(dt) 
X = (mu-0.5*sigma**2)*t + sigma*W 
S = S0*np.exp(X)
plt.plot(t,S)
plt.ylabel('Movimiento Geometrico Browniano')
plt.xlabel('Tiempo')
plt.show()