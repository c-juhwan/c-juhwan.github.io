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
y0 = threshold_function(x)
y1 = sigmoid(x)
y2 = relu(x)
y3 = tanh(x)

plt.plot(x, y0, color='black')
plt.plot(x, y1, color='blue')
plt.plot(x, y2, color='red')
plt.plot(x, y3, color='green')

plt.ylim([-1.5, 5])
plt.grid(True)
plt.show()