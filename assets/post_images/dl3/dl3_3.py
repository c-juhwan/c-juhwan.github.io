import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Perceptron:
    def __init__(self, w, bias=-0.5):
        self.weight = w
        self.bias = bias

    def affine(self, x):
        self.sum = np.sum(self.weight * x) + self.bias
        return self.sum
    
    def activation(self):
        # Sigmoid activation
        return sigmoid(self.sum)

    def work(self, x):
        self.affine(x)
        return self.activation()

draw_x = np.linspace(-5.0, 5.0, 1000)
draw_y = sigmoid(draw_x)

x = np.array([
    [-3, -1, -2],
    [0.3, 0.4, -0.5],
    [-0.4, 0.2, 0.7],
    [0.1, 0.5, 0.9],
    [0.6, 0.7, 0.9],
    [1.2, 0.4, 1.5],
    [2, 2, 2]
])
w = np.array([0.3, 0.2, 0.5])

neuron = Perceptron(w)
plt.plot(draw_x, draw_y)

for each_x in x:
    affine_point = np.round(neuron.affine(each_x), 3)
    point_y = np.round(neuron.work(each_x), 3)
    print(str(each_x) + " -> " + str(affine_point) + " -> " + str(point_y))
    plt.scatter(affine_point, point_y, color='red')

plt.ylim([-0.5, 1.5])
plt.grid(True)
plt.show()