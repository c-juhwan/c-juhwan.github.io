import numpy as np
import matplotlib.pyplot as plt

def threshold_function(x, threshold=0):
    return np.array(x>threshold, dtype=np.int32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.arange(-10, 10, 0.01)
y = tanh(x)

plt.plot(x, y)

plt.grid(True)
plt.show()