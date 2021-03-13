import numpy as np

def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))

input_a = np.array([-3, -1.5, 0.3, 0.6, 1, 1.8, 3])

print(np.round(softmax(input_a), 3))