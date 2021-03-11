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


AND_weight = np.array([0.5, 0.5])
AND_threshold = 0.7
OR_weight = np.array([0.5, 0.5])
OR_threshold = 0.3
NAND_weight = np.array([-0.5, -0.5])
NAND_threshold = -0.7

AND = Perceptron(AND_weight, AND_threshold)
OR = Perceptron(OR_weight, OR_threshold)
NAND = Perceptron(NAND_weight, NAND_threshold)

for input in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    AND_output = AND.work(np.array([input[0], input[1]]))
    OR_output =  OR.work(np.array([input[0], input[1]]))
    NAND_output = NAND.work(np.array([input[0], input[1]]))

    print(str(input) + ": " + "AND = " + str(AND_output) + ", OR = " + str(OR_output) + ", NAND = " + str(NAND_output))

