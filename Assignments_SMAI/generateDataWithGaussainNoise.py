import numpy as np
import matplotlib.pylab as plt
import math as math

def corr_vars( start=-10, stop=10, step=0.5, mu=0, sigma=3, func=lambda x: x ):
    # Generate x
    x = np.arange(start, stop, step)

    # Generate random noise
    e = np.random.normal(mu, sigma, x.size)

    # Generate y values as y = func(x) + e
    y = np.zeros(x.size)

    for ind in range(x.size):
        y[ind] = func(x[ind]) + e[ind]

    return (x,y)

np.random.seed(2)

#(x0,y0) = corr_vars(sigma=3)
#(x1,y1) = corr_vars(sigma=3, func=lambda x: 2*math.pi*math.sin(x))
(x1,y1) = corr_vars(start = -25, stop= 25 ,sigma=7, mu=0, func=lambda x: math.sin(x))
(x2,y2) = corr_vars(start = -2500, stop= 2500 ,sigma=40, mu=0, func=lambda x: math.sin(x))

f, axarr = plt.subplots(2, sharex=True, figsize=(7,7))

#axarr[0].scatter(x0, y0)
#axarr[0].plot(x0, x0, color='r')
#axarr[0].set_title('y = x + e')
#axarr[0].grid(True)
#
axarr[0].scatter(x1, y1)
axarr[0].plot(x1, np.sin(x1), color='r')
axarr[0].set_title('y = 2*π*sin(x) + e')
axarr[0].grid(True)


axarr[1].scatter(x2, y2)
axarr[1].plot(x2, np.sin(x2), color='r')
axarr[1].set_title('y = sin(x) + e')
axarr[1].grid(True)
plt.show()
from sklearn import mixture
mixture.GMM