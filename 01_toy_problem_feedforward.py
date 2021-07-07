import numpy as np
import math
import logging

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('datafile', '', 'The input data file in CSV format.')
flags.DEFINE_string('loadfile', '', 'The file to read weights from. If not given, use random weights')
flags.DEFINE_string('savefile', '', 'The file to store the network weights. If not given, nothing is saved')
flags.DEFINE_float('learningrate', 0.01, 'The learning rate')
flags.DEFINE_boolean('train', False, 'Whether to train the model or only evaluate')
flags.DEFINE_multi_integer('dim', [4,3,2], 'The dimensions of the NN, only used if no loadfile is given.')
flags.DEFINE_integer('maxruns', 1000, 'The number of runs to execute.')
flags.DEFINE_string('loglevel', 'INFO', 'logging level')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    def __init__(self, layers):
        self.activation = sigmoid
        self.layers = layers
        logger.info("New NN with shape: %s", [l.state.size for l in self.layers])

    @staticmethod
    def WithRandomWeights(dimensions):
        lastDim = dimensions[0]
        layers = []
        for dimension in dimensions[1:]:
            layers.append(Layer(np.random.rand(dimension, lastDim), np.zeros(lastDim)))
            lastDim = dimension
        return NN(layers)

    @staticmethod
    def WithGivenWeights(weights):
        layers = []
        for newweights in weights:
            layers.append(Layer(newweights, np.zeros(newweights.shape[1])))
        return NN(layers)

    @staticmethod
    def LoadFromFile(file):
        """Loads a NN from file."""
        npzfile = np.load(file)
        layers = []
        for weights in sorted(npzfile.files):
            logger.info("loading layer %s, %s", weights, npzfile[weights])
            layers.append(npzfile[weights])
        return NN.WithGivenWeights(layers)

    def Store(self, file):
        """Stores a NN to file."""
        arrays = [layer.weights for layer in self.layers]
        np.savez(file, *arrays)        


class ForwardEvaluator:
    def __init__(self, optimizer = None):
        self.optimizer = optimizer

    """Forward-evaluates a neural network."""
    def EvalLoop(self, nn, maxruns, input):
        count = 0
        correct = 0
        while count < maxruns:
            for line in input:
                if count == maxruns:
                    break
                target = int(line[0])
                input = np.asfarray(line[1:]) / 255
                output = self.Evaluate(nn, input)
                logger.debug("Input: %s Output: %s", input, output)
                count += 1

                targetVector = np.zeros(output.size)
                targetVector[target] = 1

                outputScalar = np.where(output == max(output))[0][0]
                logger.debug("Output: %s", output)
                logger.debug("Output scalar: %s", outputScalar)

                if target == outputScalar:
                    correct += 1
                if self.optimizer:
                    self.optimizer.Optimize(nn, output, targetVector)

        logger.info("Success rate: %d of %d (%f%%)", correct, count, correct * 100.0 / count)

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


def main(argv):
    logger.setLevel(FLAGS.loglevel)
    # nn = NN.WithRandomWeights([4,3,2])
    # nn = NN.WithGivenWeights([np.array([[-0.3, -0.7, -0.9,-0.9],[-1,-0.6,-0.6, -0.6],[0.8, 0.5, 0.7, 0.8]]),
    #                np.array([[2.6, 2.1, -1.2],[-2.3, -2.3, 1.1]])])
    if FLAGS.loadfile:
        nn = NN.LoadFromFile(FLAGS.loadfile)
    else:
        nn = NN.WithRandomWeights(FLAGS.dim)
    

    if FLAGS.train:
        evaluator = ForwardEvaluator(GradientDescentOptimizer(FLAGS.learningrate))
    else:
        evaluator = ForwardEvaluator()
    
    evaluator.EvalLoop(nn, FLAGS.maxruns, np.loadtxt(FLAGS.datafile, delimiter=","))

    if FLAGS.savefile:
        nn.Store(FLAGS.savefile)

if __name__ == '__main__':
  app.run(main)
