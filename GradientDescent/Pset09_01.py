import numpy as np
import matplotlib.pyplot as plt
X=np.linspace(0,50,1000)
Y=1.5*X+2
plt.scatter(X,Y,label='y=1.5x+2',marker='.')
plt.title('Linear data')
plt.legend()
plt.show()
noise=np.random.normal(0,10,1000)
print(noise)
Y=Y+noise
plt.scatter(X,Y,label='y=1.5x+c+N(\mu=0,\sigma=1)')
plt.legend()
plt.title('Noise data points')
plt.show()

