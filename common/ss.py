import matplotlib.pyplot as plt

value  = [(1, 3005),(2,6004), (3, 9003), (4, 12005), (5, 15003)]
plt.rcParams.update({'font.size': 22})
plt.plot(value)
plt.xlabel("Number of nodes")
plt.ylabel("Time in leader election in mills")
plt.show()

