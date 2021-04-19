# Bryce Allen
# Applied Machine Learning
# Project 3

import numpy as np
from sklearn.datasets import load_breast_cancer

input_ft = np.array([[0,0],[0,1],[1,0],[1,1]])
#input_ft = load_breast_cancer()
#input_ft.target[[10,50,85]]
#print (input_ft.target)
print (input_ft.shape)
input_ft

target = np.array([0,1,1,1])
target = target.reshape(4,1)
print(target.shape)
target

hidden_weight = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
output_weight = np.array([[0.7],[0.8],[0.9]])

learning_rate = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sig_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(200000):
    hidden_input = np.dot(input_ft,hidden_weight)
    hidden_output = sigmoid(hidden_input)
    input_output_layer = np.dot(hidden_output, output_weight)
    output_output_layer = sigmoid(input_output_layer)
    error_output = ((1/2) * (np.power((output_output_layer - target), 2)))
    print(error_output.sum())
    derivative_error_target = output_output_layer - target
    derivative_input = sig_derivative(input_output_layer)
    derivative_output = hidden_output
    derivative_error_output = np.dot(derivative_output.T, derivative_error_target * derivative_input)

    derivative_error_input = derivative_error_target * derivative_input
    derivative_input_output = output_weight
    derivative_outputth = np.dot(derivative_error_input, derivative_input_output.T)
    derivative_input_hidden = sig_derivative(hidden_input)
    derivative_input_dwh = input_ft
    derivative_error_weight = np.dot(derivative_input_dwh.T, derivative_input_hidden * derivative_outputth)

    hidden_weight -= learning_rate * derivative_error_weight
    output_weight -= learning_rate * derivative_error_output

