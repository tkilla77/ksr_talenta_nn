import numpy as np
import math

def sigmoid(input):
    return 1 / (1 + math.exp(-input))

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.activation = sigmoid

class ForwardEvaluator:
    def Evaluate(nn, input):
        state = input
        for layer in nn.layers:
            state = np.dot(layer, state)
            state = np.array([np.activation(i) for i in input])
        return state

def main():
    nn = NN(values)
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


