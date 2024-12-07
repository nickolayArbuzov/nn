import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


x = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
    ]
)
y = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

input_size = len(x[0])
hidden_size = 4
output_size = len(y[0])

np.random.seed(1)
weights_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_output = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.01
epochs = 100000

for epoch in range(epochs):
    layer_hidden = sigmoid(np.dot(x, weights_hidden))
    layer_out = softmax(np.dot(layer_hidden, weights_output))
    error = (layer_out - y) ** 2

    layer_out_delta = (layer_out - y) / len(layer_out)
    layer_hidden_delta = layer_out_delta.dot(weights_output.T) * sigmoid_deriv(
        layer_hidden
    )

    weights_output -= learning_rate * layer_hidden.T.dot(layer_out_delta)
    weights_hidden -= learning_rate * x.T.dot(layer_hidden_delta)
    if epoch % 1000 == 0:
        print("error", error)


def predict(input):
    layer_hidden = sigmoid(np.dot(input, weights_hidden))
    layer_out = softmax(np.dot(layer_hidden, weights_output))
    print(layer_out)
    return np.argmax(layer_out)


for input in x:
    print(f"digit - {input}: ", predict(np.array([input])))
