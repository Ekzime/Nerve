from nerve.layer import Layer
import numpy as np

class Linear(Layer):
    def __init__(self, input_size, output_size):
        # create weights and bias
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        
        self.output_date_after_forward = None

    def forward(self, input_data):
        self.input = input_data
        # math: y = x @ w + b
        output = self.input @ self.weights + self.biases
        
        return output

    def backward(self, output_gradient, learning_rate):
        # gradient for weights
        grad_weights = self.input.T @ output_gradient
        
        # gradient for bias(batch sum)
        grad_bias = np.sum(output_gradient, axis=0)

        # gradient for input layer (for latest layer)
        grad_input = output_gradient @ self.weights.T

        # save weights
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_bias

        return grad_input

class ReLu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_data):
        # save input for backwards data
        self.input = input_data
        
        # apply func max(0,x)
        return np.maximum(0,self.input)

    def backward(self,output_gradient, learning_rate):
        # Deravative relu:
        # off gradient where input was negative
        # copy input grad
        input_gradient = output_gradient.copy()
        # update grad, where forward <= 0
        input_gradient[self.input <= 0] = 0
        return input_gradient
