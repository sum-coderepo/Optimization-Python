import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
# stock prices (3x  per day)
# [morning, midday, evening]
APPLE = np.array(
    [[1,5],[3,-2],[-1,-4],[-2,1]])

# midday variance
print(APPLE.mean(axis=0))
cov = np.cov(APPLE,rowvar=0)
print(cov)

w, v = LA.eig(cov)
print(w)
print(v)
origin = [0, 0]

eig_vec1 = v[:,0]
eig_vec2 = v[:,1]

print(eig_vec1)
print(eig_vec2)


# This line below plots the 2d points
#plt.scatter(np_array[:,0], np_array[:,1])

plt.quiver(*origin, *eig_vec1, color=['r'], scale=21)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=21)
plt.show()
