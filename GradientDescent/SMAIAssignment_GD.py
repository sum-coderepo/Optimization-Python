from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
cur_x = -2 # The algorithm starts at x=-2
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x: 2*x + 6 #Gradient of our function

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x #Store current x value in prev_x
    cur_x = cur_x - rate * df(prev_x) #Grad descent
    previous_step_size = abs(cur_x - prev_x) #Change in x
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nX value is",cur_x ," at current timestamp " ,datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")) #Print iterations

print("The local minimum occurs at", cur_x)
#X=np.linspace(0,50,1000)
#Y=1.5*X+2
#plt.scatter(X,Y,label='y=1.5x+2',marker='.')
#plt.title('Linear data')
#plt.legend()
#plt.show()
#noise=np.random.normal(0,10,1000)
#Y=Y+noise
#print(Y.shape)
#print(X.shape)
#plt.axis