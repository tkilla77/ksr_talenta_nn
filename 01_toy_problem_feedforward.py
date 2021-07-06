import numpy as np
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def sigmoid(input):
    """The sigmoid function."""
    return 1 / (1 + math.exp(-input))

class Layer:
    """One layer in a neural network.
    It carries its neurons' input state and its input weights.
    """
    def __init__(self, weights, state):
        self.weights = weights
        self.state = state
        assert weights.shape[1] == state.shape[0], "Expected compatible size: %s/%s" % (weights.shape, state.shape)

class NN:
    """A neural network."""
    def __init__(self):
        self.activation = sigmoid
        self.layers = []

    def SetRandomWeights(self, dimensions):
        lastDim = dimensions[0]
        self.layers = []
        for dimension in dimensions[1:]:
            self.layers.append(Layer(np.random.rand(dimension, lastDim), np.zeros(lastDim)))
            lastDim = dimension

    def SetWeights(self, weights):
        logger.info("SetWeights: old shape: %s", [l.state.size for l in self.layers])
        self.layers = []
        for newweights in weights:
            self.layers.append(Layer(newweights, np.zeros(newweights.shape[1])))
        logger.info("New shape: %s", [l.state.size for l in self.layers])

class ForwardEvaluator:
    """Forward-evaluates a neural network."""
    def Evaluate(self, nn, input):
        state = input
        for layer in nn.layers:
            layer.state = state
            state = np.dot(layer.weights, state)
            state = np.array([nn.activation(i) for i in state])
        return state

class GradientDescentOptimizer:
    """Implements gradient descent and changes weights in the NN."""
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def Cost(self, output, target):
        return target - output

    def Optimize(self, nn, output, target):
        error = self.Cost(output, target)
        
        for layer in reversed(nn.layers):
            logger.debug("Error is: %s", error)
            next_error = np.dot(layer.weights.T, error)
            term = (error * output * (1-output)).reshape(-1,1)
            logger.debug("Gradient term shape: %s", term.shape)
            state_T = layer.state.reshape(1,-1)
            logger.debug("State shape is: %s", state_T.shape)
            gradient = np.dot(term, state_T)
            logger.debug("Gradient is: %s", gradient)
            logger.debug("Increment is: %s", -1 * self.learning_rate * gradient)
            layer.weights = layer.weights + self.learning_rate * gradient
            output = layer.state
            error = next_error


class NetworkIO:
    """Loads and stores NNs"""
    @staticmethod
    def Store(file, nn):
        """Stores a NN to file."""
        np.savez(file, *nn.layers)

    @staticmethod
    def Load(file):
        """Loads a NN from file."""
        npzfile = np.load(file)
        layers = []
        for weights in npzfile.files:
            layers.append(npzfile[weights])
        nn = NN()
        nn.SetWeights(layers)
        return nn
        

def main():
    nn = NetworkIO.Load("toy_network.nn.npz")
    # nn = NN()
    # nn.SetRandomWeights([4,3,2])
    # nn.SetWeights([np.array([[-0.3, -0.7, -0.9,-0.9],[-1,-0.6,-0.6, -0.6],[0.8, 0.5, 0.7, 0.8]]),
    #                np.array([[2.6, 2.1, -1.2],[-2.3, -2.3, 1.1]])])

    evaluator = ForwardEvaluator()
    optimizer = GradientDescentOptimizer(1)
    count = 0
    correct = 0
    for line in np.loadtxt("data_toy_problem/data_dark_bright_training_20000.csv", delimiter=","):
        target = line[0]
        input = np.asfarray(line[1:]) / 255
        output = evaluator.Evaluate(nn, input)
        logger.debug("Input: %s Output: %s", input, output)
        count += 1
        if target == (output[0] < output[1]):
            correct += 1
        if target == 0:
            targetVector = np.array([1,0])
        else:
            targetVector = np.array([0,1])
        optimizer.Optimize(nn, output, targetVector)


    logger.info("Success rate: %d of %d (%f%%)", correct, count, correct * 100.0 / count)
    NetworkIO.Store("toy_network.nn", nn)

main()