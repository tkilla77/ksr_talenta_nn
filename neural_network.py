"""
Usage:

Training: train a new network from scratch

$ python neural_network.py --dim 784 --dim 100 --dim 50 --dim 10 \
    --savefile mnist_best.npz \
    --datafile data_mnist/mnist_train.csv \
    --train --learningrate 0.02 --maxruns 400000


Eval:
$ python neural_network.py \
    --loadfile mnist_best.npz \
    --datafile data_mnist/mnist_test.csv \
    --maxruns 4000

"""
import itertools
import numpy as np
import logging
from PIL import Image

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
flags.DEFINE_integer('reportingBatchSize', 1000, 'The number of evaluations after which to report.')
flags.DEFINE_string('loglevel', 'INFO', 'logging level')
flags.DEFINE_integer('rotate', '5', 'randomly rotate images by +/- so many degrees')
flags.DEFINE_integer('translate', '5', 'randomly move images by +/- so many pixels in both axes')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def sigmoid(input):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-input))

class Layer:
    """One layer in a neural network.
    It carries its neurons' input state and its output weights.
    """
    def __init__(self, weights, activation=sigmoid):
        self.weights = weights
        self.state = None
        self.activation = activation
    
    def __str__(self) -> str:
        return "FC: %s" % str(self.weights.shape[::-1])

    def Evaluate(self, input):
        """A feed-forward pass in this layer, returning the layer's output."""
        assert self.weights.shape[1] == input.shape[0], "Expected compatible size: %s/%s" % (self.weights.shape, input.shape)
        # Store the input for backprop.
        self.state = input

        return self.activation(np.dot(self.weights, input))

class NN:
    """A neural network."""
    def __init__(self, layers):
        self.layers = layers
        logger.info("New NN with shape: %s", [str(l) for l in self.layers])

    @staticmethod
    def WithRandomWeights(dimensions):
        lastDim = dimensions[0]
        layers = []
        for dimension in dimensions[1:]:
            layers.append(Layer(np.random.rand(dimension, lastDim) - 0.5))
            lastDim = dimension
        return NN(layers)

    @staticmethod
    def WithGivenWeights(weights):
        layers = []
        for newweights in weights:
            layers.append(Layer(newweights))
        return NN(layers)

    @staticmethod
    def LoadFromFile(file):
        """Loads a NN from file."""
        npzfile = np.load(file)
        layers = []
        for weights in sorted(npzfile.files):
            logger.debug("loading layer %s, %s", weights, npzfile[weights])
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
    def EvalLoop(self, nn, maxruns, input, reportingBatchSize = 100):
        count = 0
        correct = 0
        batchCorrect = 0
        batchError = 0.0

        for line in itertools.cycle(input):
            if count == maxruns:
                break

            count += 1
            target = line['target']
            input = line['input']
            input = random_transform(input, self.optimizer is None)
            output = self.Evaluate(nn, input)

            # Choose an arbitrary index with the highest activation as the output.
            outputScalar = np.argmax(output)

            if target == outputScalar:
                correct += 1
            if self.optimizer:
                # Target value for classification problem is a one hot vector.
                targetVector = np.zeros(output.size)
                targetVector[target] = 1
                targetVector = targetVector.reshape((-1, 1))
                batchError += self.optimizer.Optimize(nn, output, targetVector)
            
            # Report a few quality stats each batch.
            if count % reportingBatchSize == 0:
                batchSuccess = correct - batchCorrect
                batchRate = batchSuccess / reportingBatchSize
                overallRate = correct / count
                avgError = batchError / reportingBatchSize
                logger.info(f"Batch ({count / reportingBatchSize:n}): Avg error / Batch Acc / Overall Acc: {avgError:.3} / {batchRate:.1%} / {overallRate:.1%}")
                batchError = 0.0
                batchCorrect = correct

        logger.info("Success rate: %d of %d (%f%%)", correct, count, correct * 100.0 / count)

    def Evaluate(self, nn, input):
        """A single feed-forward pass."""
        state = input
        for layer in nn.layers:
            state = layer.Evaluate(state)
        return state

class GradientDescentOptimizer:
    """Implements gradient descent and changes weights in the NN."""
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def Cost(self, output, target):
        return target - output

    def Optimize(self, nn, output, target):
        error = self.Cost(output, target)
        output_error = np.linalg.norm(error)
        
        for layer in reversed(nn.layers):
            next_error = np.dot(layer.weights.T, error)

            # Update this layer's weights.
            gradient = np.dot(error * output * (1-output), layer.state.T)
            layer.weights = layer.weights + self.learning_rate * gradient

            # Move one network layer to the left.
            output = layer.state
            error = next_error

        return output_error

def readCsvLines255(filename):
    """
    Reads CSV lines and divides input values by 255.
    Returns a dictionary with entries 'target' and 'input'.
    """
    for row in open(filename, "r"):
        split = row.split(",")
        target = int(split[0])
        input =  np.asarray(split[1:], dtype=np.uint8)
        yield {'target': target, 'input': input}

def random_transform(pixels, eval=True, degrees=None, translation=None):
    """
    Randomly rotates the image by [-degrees,degrees] and move by translation pixels in a random direction.
    New pixels are filled with black.
    """
    if eval:
        return pixels.reshape(-1,1) / 255
    
    if degrees is None:
        degrees = FLAGS.rotate
    if translation is None:
        translation = FLAGS.translate
        
    pixels = pixels.reshape(28,28)
    # mode 'L' means 8bit grayscale
    with Image.fromarray(pixels, mode="L") as image:
        angle = np.random.randint(-degrees, degrees+1)
        x = np.random.randint(-translation, translation+1)
        y = np.random.randint(-translation, translation+1)
        move = (x,y)
        transformed = image.rotate(angle=angle, resample=Image.Resampling.NEAREST, translate=move, fillcolor=0)
        pixel_data = transformed.getdata()
        return np.array(pixel_data, dtype=np.float32).reshape(-1,1) / 255

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
    
    evaluator.EvalLoop(nn, FLAGS.maxruns, readCsvLines255(FLAGS.datafile), FLAGS.reportingBatchSize)

    if FLAGS.savefile and FLAGS.train:
        nn.Store(FLAGS.savefile)

if __name__ == '__main__':
  app.run(main)
