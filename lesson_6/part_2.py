import numpy as np

np.random.seed(100)


def relu(x):
    return (x > 0) * x


def reluderiv(x):
    return x > 0


input = np.array([[15, 10], [15, 15], [15, 20], [25, 10]])

true_predictions = np.array([[10, 20, 15, 20]]).T
layer_in_size = len(input[0])
layer_hidden_1_size = 3
layer_out_size = len(true_predictions[0])

weights_in_hidden_1 = 2 * np.random.random((layer_in_size, layer_hidden_1_size)) - 1
print("weights_in_hidden_1", weights_in_hidden_1)
weights_hidden_1_out = 2 * np.random.random((layer_hidden_1_size, layer_out_size)) - 1
print("weights_hidden_1_out", weights_hidden_1_out)

learning_rate = 0.00001
num_epochs = 1000

for i in range(num_epochs):
    layer_out_error = 0
    for j in range(len(input)):
        layer_in = input[j : j + 1]
        print("layer_in", layer_in)
        layer_hidden_1 = relu(np.dot(layer_in, weights_in_hidden_1))
        print("layer_hidden_1", layer_hidden_1)
        layer_out = np.dot(layer_hidden_1, weights_hidden_1_out)
        layer_out_error += np.sum((layer_out - true_predictions[j : j + 1]) ** 2)
        layer_out_delta = layer_out - true_predictions[j : j + 1]
        layer_hidden_1_delta = layer_out_delta.dot(weights_hidden_1_out.T) * reluderiv(
            layer_hidden_1
        )
        weights_hidden_1_out -= learning_rate * layer_hidden_1.T.dot(layer_out_delta)
        weights_in_hidden_1 -= learning_rate * layer_in.T.dot(layer_hidden_1_delta)
        print("Predictions: %s, True_Predictions: %s" % (layer_out, true_predictions))
    print("layer_out_error", layer_out_error)
