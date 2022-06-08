import math
from random import random

from jinja2 import Undefined


class Neuron:
    # Initialize the neuron with custom number of inputs
    def __init__(self, inputs, weights=Undefined, bias=Undefined, activation_function="sigmoid"):
        self.activation_function = activation_function
        # self.groundTruth = inputs[len(inputs) - 1]
        # del inputs[-1]
        self.inputs = inputs
        if weights is Undefined:
            self.weights = [random.uniform(0, 1) for i in range(len(self.inputs))]
        else:
            self.weights = weights
        if bias is Undefined:
            self.bias = 0
        else:
            self.bias = bias

    def train(self):
        if self.activation_function == "sigmoid":
            s = self.bias + sum([i * j for i, j in zip(self.weights, self.inputs)])
            y = 1 / (1 + math.exp(-s))
        elif self.activation_function == "relu":
            y = max(0, self.bias + sum([i * j for i, j in zip(self.weights, self.inputs)]))
        elif self.activation_function == "tanh":
            y = math.tanh(self.bias + sum([i * j for i, j in zip(self.weights, self.inputs)]))
        elif self.activation_function == "linear":
            y = self.bias + sum([i * j for i, j in zip(self.weights, self.inputs)])
        return y

    def predict(self):
        result = {"right": 0, "wrong": 0}
        y = self.train()
        if y > 0 and self.groundTruth>0 :
            # print("Good")
            result["right"] += 1
        elif y > 0 and self.groundTruth == 0:
            # print("Bad")
            result["wrong"] += 1
        elif y < 0 and self.groundTruth > 0:
            # print("Bad")
            result["wrong"] += 1
        elif y < 0 and self.groundTruth == 0:
            # print("Good")
            result["right"] += 1
        return result

    # backpropagation modefying weights and bias
    def backpropagation(self, learning_rate):
        train_result = self.train()
        derivative_bias = learning_rate * (
                (train_result - self.groundTruth) * (train_result - train_result ** 2) * train_result)
        for i in range(len(self.inputs)):
            self.weights[i] = self.weights[i] - derivative_bias
        # print("new weights: ", self.weights)
        return self.weights

    def get_error(self):
        return 0.5 * (self.groundTruth - self.train()) ** 2
