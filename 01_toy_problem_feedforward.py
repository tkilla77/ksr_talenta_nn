import numpy as np
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def sigmoid(input):
    return 1 / (1 + math.exp(-input))

class NN:
    def __init__(self, layers):
        lastDim = layers[0]
        self.layers = []
        for dimension in layers[1:]:
            self.layers.append(np.random.rand(dimension, lastDim))
            lastDim = dimension
        self.activation = sigmoid

    def SetWeights(self, weights):
        logger.debug("SetWeights: old shape: %s", [l.shape for l in self.layers])
        self.layers = weights
        logger.debug("New shape: %s", [l.shape for l in self.layers])

class ForwardEvaluator:
    def Evaluate(self, nn, input):
        state = input
        for layer in nn.layers:
            state = np.dot(layer, state)
            state = np.array([nn.activation(i) for i in state])
        return state

def parseLine(line):
    values = line.split(",")
    target = int(values[0])
    input = np.array([val / 255 for val in values[1:]])
    return target, input

def main():
    nn = NN([4,3,2])
    nn.SetWeights([np.array([[-0.3, -0.7, -0.9,-0.9],[-1,-0.6,-0.6, -0.6],[0.8, 0.5, 0.7, 0.8]]),
                   np.array([[2.6, 2.1, -1.2],[-2.3, -2.3, 1.1]])])

    evaluator = ForwardEvaluator()
    count = 0
    correct = 0
    for line in np.loadtxt("data_toy_problem/data_dark_bright_test_4000.csv", delimiter=","):
        target = line[0]
        input = np.array([val / 255 for val in line[1:]])
        output = evaluator.Evaluate(nn, input)
        logger.debug("Input: %s Output: %s", input, output)
        count += 1
        if target == (output[0] < output[1]):
            correct += 1
            logger.debug("Result: %s, target: %s", output[1], target)
        else:
            logger.debug("Result: %s, target: %s", output[0], target)

    logger.info("Success rate: %d of %d (%d%%)", correct, count, correct * 100 / count)

main()

