from Neuron import Neuron


class InputLayer:
    # intialize the input layer with exactly one input for the neuron
    def __init__(self, number_of_neurons, inputs):
        # create neurons with the input and the activation function
        self.number_of_neurons = number_of_neurons
        self.neurons = [Neuron(inputs=inputs) for i in range(number_of_neurons)]
