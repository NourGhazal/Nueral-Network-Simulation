from Neuron import Neuron


class OutputLayer:
    # initialize the output layer with the number of neurons
    def __init__(self, number_of_neurons, activation_function, inputs):
        self.number_of_neurons = number_of_neurons
        self.activation_function = activation_function
        self.neurons = [
            Neuron(inputs=inputs, activation_function=self.activation_function)
            for i in range(self.number_of_neurons)]
