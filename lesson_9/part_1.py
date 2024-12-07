import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = len(x[0])
hidden_size = 4
output_size = len(y[0])

np.random.seed(1)
weights_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_output = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 100000

for epoch in range(epochs):
    layer_hidden = sigmoid(np.dot(x, weights_hidden))
    layer_out = sigmoid(np.dot(layer_hidden, weights_output))
    error = (layer_out - y) ** 2

    layer_out_delta = (layer_out - y) * sigmoid_deriv(layer_out)
    layer_hidden_delta = layer_out_delta.dot(weights_output.T) * sigmoid_deriv(
        layer_hidden
    )

    weights_output -= learning_rate * layer_hidden.T.dot(layer_out_delta)
    weights_hidden -= learning_rate * x.T.dot(layer_hidden_delta)

new_input = np.array([[0, 1]])
layer_hidden = sigmoid(np.dot(new_input, weights_hidden))
layer_out = sigmoid(np.dot(layer_hidden, weights_output))

print(layer_out)
