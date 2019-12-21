import matplotlib.pyplot as plt
import numpy as np

X = np.random.randint(1, 200, 20)
y = np.random.randint(1, 200, 20)
X2 = np.random.randint(1, 200, 20)
y2 = np.random.randint(1, 200, 20)
print(X)
print(y)
plt.scatter(X,y, label = "first", c = "b" , s = 10, marker = "*")
plt.scatter(X2, y2, label = "second" , c = "orange" , s = 250 , marker = "s")
plt.xlabel("X", fontsize = 10)
plt.ylabel("Y", fontsize = 10)
plt.xticks(size = 10)
plt.title("Cool scatter flat ! \n hi ", fontsize = 18)
plt.show()