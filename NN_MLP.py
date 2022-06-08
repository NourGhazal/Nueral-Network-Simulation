from turtle import shape
import cv2
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
import random


# names:
# Noureldin Ghazal 43-15747 noureldin.ghazal@student.guc.edu.eg
# Kareem Fathy 43-6912 Kareem.abdelhady@student.guc.edu.eg
# Mohamed Ahmed 43-16620 mohamed.alekhsasy@student.guc.edu.eg
# Mahmoud Yehia 43-12312 Mahmoud.yehia@student.guc.edu.eg

class NN_MLP:
    # use class neuron, input layer, hidden layer and output layer to create a neural network
    def __init__(self, input_layer, hidden_layers, output_layer, learning_rate):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.weights = []

    def train(self, input_image,ground_truth):
        inputs = input_image
        final_outputs = []
        for i in range(len(inputs)):
            hidden_layer_inputs = []
            for j in range(self.input_layer.number_of_neurons):
                if(inputs[i].shape==()):
                    self.input_layer.neurons[j].inputs = [inputs[i]]
                    hidden_layer_inputs.append(inputs[i])
                else:
                    self.input_layer.neurons[j].inputs = [inputs[i][j]]
                    hidden_layer_inputs.append(inputs[i][j])
                self.input_layer.neurons[j].weights= [random.uniform(0, 1) for i in range(self.input_layer.number_of_neurons)]
                self.input_layer.neurons[j].groundTruth = ground_truth[i]
            hidden_layer_outputs = []
            for j in range(self.hidden_layers[0].number_of_neurons):
                self.hidden_layers[0].neurons[j].inputs = hidden_layer_inputs
                self.hidden_layers[0].neurons[j].weights= [random.uniform(0, 1) for i in range(len(hidden_layer_inputs))]
                self.hidden_layers[0].neurons[j].groundTruth = ground_truth[i]
                hidden_layer_outputs.append(self.hidden_layers[0].neurons[j].train())
                self.hidden_layers[0].neurons[j].weights = self.hidden_layers[0].neurons[j].backpropagation(self.learning_rate)
            for j in range(1, len(self.hidden_layers)):
                hidden_layer_inputs = hidden_layer_outputs
                hidden_layer_outputs = []
                for k in range(self.hidden_layers[j].number_of_neurons):
                    self.hidden_layers[j].neurons[k].inputs = hidden_layer_inputs
                    self.hidden_layers[j].neurons[k].weights= [random.uniform(0, 1) for i in range(len(hidden_layer_inputs))]
                    self.hidden_layers[j].neurons[k].groundTruth = ground_truth[i]
                    hidden_layer_outputs.append(self.hidden_layers[j].neurons[k].train())
                    self.hidden_layers[j].neurons[k].weights = self.hidden_layers[j].neurons[k].backpropagation(self.learning_rate)
            for j in range(self.output_layer.number_of_neurons):
                self.output_layer.neurons[j].inputs = hidden_layer_outputs
                self.output_layer.neurons[j].weights= [random.uniform(0, 1) for i in range(len(hidden_layer_outputs))]
                self.output_layer.neurons[j].groundTruth = ground_truth[i]
                final_outputs.append(self.output_layer.neurons[j].train())
        return final_outputs

    def test(self, test_image,ground_truth):
        inputs = test_image
        final_result = {"right": 0, "wrong": 0}
        for i in range(inputs.shape[0]):
            hidden_layer_inputs = []
            for j in range(self.input_layer.number_of_neurons):
                if(inputs[i].shape==()):
                    self.input_layer.neurons[j].inputs = [inputs[i]]
                    hidden_layer_inputs.append(inputs[i])
                else:
                    self.input_layer.neurons[j].inputs = [inputs[i][j]]
                    hidden_layer_inputs.append(inputs[i][j])
                self.input_layer.neurons[j].weights= [random.uniform(0, 1) for i in range(self.input_layer.number_of_neurons)]
                self.input_layer.neurons[j].groundTruth = ground_truth[i]
            hidden_layer_outputs = []
            for j in range(self.hidden_layers[0].number_of_neurons):
                self.hidden_layers[0].neurons[j].inputs = hidden_layer_inputs
                self.hidden_layers[0].neurons[j].groundTruth = ground_truth[i]
                hidden_layer_outputs.append(self.hidden_layers[0].neurons[j].train())
            for j in range(1, len(self.hidden_layers)):
                hidden_layer_inputs = hidden_layer_outputs
                hidden_layer_outputs = []
                for k in range(self.hidden_layers[j].number_of_neurons):
                    self.hidden_layers[j].neurons[k].inputs = hidden_layer_inputs
                    self.hidden_layers[j].neurons[k].groundTruth = ground_truth[i]
                    hidden_layer_outputs.append(self.hidden_layers[j].neurons[k].train())
            for j in range(self.output_layer.number_of_neurons):
                self.output_layer.neurons[j].inputs = hidden_layer_outputs
                self.output_layer.neurons[j].groundTruth = ground_truth[i]
                result = self.output_layer.neurons[j].predict()
                final_result["right"] += result["right"]
                final_result["wrong"] += result["wrong"]
        return final_result
# main method
if __name__ == "__main__":
    im_gray = cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    full_image = im_bw.flatten()
    training_data = full_image[:int(len(full_image) * 0.8)]
    testing_data = full_image[int(len(full_image) * 0.8):]
    input_layer = InputLayer(1,[])
    number_of_hidden_layers = 1
    hidden_layers= []
    for i in range(number_of_hidden_layers):
        hidden_layers.append(HiddenLayer(2,"relu",[]))
    output_layer = OutputLayer(1,"sigmoid",[])
    nn = NN_MLP(input_layer, hidden_layers, output_layer, 1)
    train_output = nn.train(training_data, training_data)
    test_output = nn.test(testing_data,testing_data)
    total = test_output["right"] + test_output["wrong"]
    print("Gray scale image")
    print("Accuracy: " + str(test_output["right"] / total))
    print("Right: " + str(test_output["right"]))
    print("Wrong: " + str(test_output["wrong"]))

    # RGB 
    img_rgb = cv2.imread('MLP_DoublethrsholdingExample.png')
    full_image = img_rgb.reshape(img_rgb.shape[0] * img_rgb.shape[1], 3)
    im_gray = cv2.imread('MLP_DoublethrsholdingExample.png', cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = im_bw.flatten()
    training_data = full_image[:int(len(full_image) * 0.8)]
    testing_data = full_image[int(len(full_image) * 0.8):]
    input_layer = InputLayer(3,[])
    number_of_hidden_layers = 2
    hidden_layers= []
    hidden_layers.append(HiddenLayer(512,"relu",[]))
    hidden_layers.append(HiddenLayer(128,"relu",[]))
    output_layer = OutputLayer(1,"sigmoid",[])
    nn = NN_MLP(input_layer, hidden_layers, output_layer, 1)
    train_output = nn.train(training_data, im_bw)
    test_output = nn.test(testing_data,im_bw)
    total = test_output["right"] + test_output["wrong"]
    print("RGB image")
    print("Accuracy: " + str(test_output["right"] / total))
    print("Right: " + str(test_output["right"]))
    print("Wrong: " + str(test_output["wrong"]))

    


     
     

    
    