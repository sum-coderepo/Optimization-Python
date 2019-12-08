import numpy as np
import math
import matplotlib.pyplot as plt

#t = linspace(0, 2*math.pi, 400)
#a = sin(t)
#b = cos(t)
#c = a + b
#
#plt.plot(t, a, 'r') # plotting t, a separately
#plt.plot(t, b, 'b') # plotting t, b separately
#plt.plot(t, c, 'g') # plotting t, c separately
#plt.show()
X = 2 * np.random.rand(100,1)
print(X)
print(X.shape)
X=np.linspace(0,50,1000)
print(X.shape)
X=np.random.randint(0,50,1000)
print(X)
print(X.shape)