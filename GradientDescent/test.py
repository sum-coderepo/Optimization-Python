from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def f(a,b):
    return a**2 + b**2

fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
#plt.hold(True)
a = np.arange(-2, 2, 0.25)
b = np.arange(-2, 2, 0.25)
a, b = np.meshgrid(a, b)
c = f(a,b)
surf = ax.plot_surface(a, b, c, rstride=1, cstride=1, alpha=0.3,
                       linewidth=0, antialiased=False,cmap='rainbow')
ax.set_zlim(-0.01, 8.01)

def gradient_descent(theta0, iters, alpha):
    history = [theta0] # to store all thetas
    theta = theta0     # initial values for thetas
    # main loop by iterations:
    for i in range(iters):
        # gradient is [2x, 2y]:
        gradient = [2.0*x for x in theta]
        # update parameters:
        theta = [a - alpha*b for a,b in zip(theta, gradient)]
        history.append(theta)
    return history

history = gradient_descent(theta0 = [-1.8, 1.6], iters = 30, alpha = 0.03)

fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
#plt.hold(True)
a = np.arange(-2, 2, 0.25)
b = np.arange(-2, 2, 0.25)
a, b = np.meshgrid(a, b)
c = f(a,b)
surf = ax.plot_surface(a, b, c, rstride=1, cstride=1, alpha=0.3,
                       linewidth=0, antialiased=False)
ax.set_zlim(-0.01, 8.01)

a = np.array([x[0] for x in history])
b = np.array([x[1] for x in history])
c = f(a,b)
ax.scatter(a, b, c, color="r");

plt.show()