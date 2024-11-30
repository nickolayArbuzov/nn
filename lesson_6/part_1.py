import numpy as np


def relu(x):
    return (x > 0) * x


input = np.array([[15, 10], [15, 15], [15, 20], [25, 10]])

true_predictions = np.array([[10, 20, 15, 20]]).T
layer_in_size = len(input[0])
layer_hidden_1_size = 3
layer_out_size = len(true_predictions[0])

weights_in_hidden_1 = 2 * np.random.random((layer_in_size, layer_hidden_1_size)) - 1
print("weights_in_hidden_1", weights_in_hidden_1)
weights_hidden_1_out = 2 * np.random.random((layer_hidden_1_size, layer_out_size)) - 1
print("weights_hidden_1_out", weights_hidden_1_out)

layer_in = input[0]
layer_hidden_1 = relu(np.dot(layer_in, weights_in_hidden_1))
print("layer_hidden_1", layer_hidden_1)
layer_out = np.dot(layer_hidden_1, weights_hidden_1_out)
print("layer_out", layer_out)
