import numpy as np

class Perceptron:
    def __init__(self, w, threshold=0.5):
        self.weight = w
        self.threshold = threshold

    def work(self, x):
        sum = np.sum(self.weight * x)

        if sum <= self.threshold:
            return 0
        else: # sum > self.threshold
            return 1

def MLP_XOR(x):
    AND_weight = np.array([0.5, 0.5])
    AND_threshold = 0.7
    OR_weight = np.array([0.5, 0.5])
    OR_threshold = 0.3
    NAND_weight = np.array([-0.5, -0.5])
    NAND_threshold = -0.7

    AND = Perceptron(AND_weight, AND_threshold)
    OR = Perceptron(OR_weight, OR_threshold)
    NAND = Perceptron(NAND_weight, NAND_threshold)

    # x = input layer
    # h1, h2 = hidden layer
    # y = output layer
    h1 = NAND.work(x)
    h2 = OR.work(x)
    y = AND.work(np.array([h1, h2]))

    return y

for input in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    XOR_output = MLP_XOR(np.array([input[0], input[1]]))

    print(str(input) + ": " + "XOR = " + str(XOR_output))

