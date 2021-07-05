import numpy as np
import math

def sigmoid(input):
    return 1 / (1 + math.exp(-input))

class NN:
    def __init__(self, layers):
        lastDim = layers[0]
        self.layers = []
        for dimension in layers[1:]:
            self.layers.append(np.random.rand(lastDim, dimension))
            lastDim = dimension
        self.activation = sigmoid

    def SetWeights(weights):
        self.layers = weights

class ForwardEvaluator:
    def Evaluate(nn, input):
        state = input
        for layer in nn.layers:
            state = np.dot(layer, state)
            state = np.array([np.activation(i) for i in input])
        return state

def readInputs(file):
    for line in file.readlines():
        return line.split(",")

def main():
    nn = NN([4,3,2])
    nn.SetWeights([np.array([[−0.3, −0.7, −0.9,−0.9],[−1,−0.6,−0.6, −0.6],[0.8, 0.5, 0.7, 0.8]]),
                   np.array([[2.6, 2.1, −1.2],[−2.3, −2.3, 1.1]]))
    evaluator = ForwardEvaluator()
    count = 0
    correct = 0
    for input in readInputs():
        output = evaluator.Evaluate(nn, input[1:])
        target = input[0]
        count += 1
        if target == output[0] > output[1]:
            correct += 1

    print("Success rate: % %%", correct / count)

main()

