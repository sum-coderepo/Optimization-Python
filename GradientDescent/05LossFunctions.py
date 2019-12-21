import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def mse(true, pred):
    """
    true: array of true values
    pred: array of predicted values

    returns: mean square error loss
    """

    return np.sum((true - pred)**2)

fig, ax1 = plt.subplots(1,1, figsize = (7,5))

# array of same target value 10000 times
target = np.repeat(100, 10000)
pred = np.arange(-10000,10000, 2)

loss_mse = [mse(target[i], pred[i]) for i in range(len(pred))]

# plot
ax1.plot(pred, loss_mse)
ax1.set_xlabel('Predictions')
ax1.set_ylabel('Loss')
ax1.set_title("MSE Loss vs. Predictions")

fig.tight_layout()
plt.show()