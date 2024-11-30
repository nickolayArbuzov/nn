import numpy as np


def nn(input, weights):
    return input.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


input = np.array([150, 40])
weigths = np.array([0.2, 0.3])

true_prediction = 1
learning_rate = 0.00001

for i in range(300):
    prediction = nn(input, weigths)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weigths, error))
    delta = (prediction - true_prediction) * input * learning_rate
    delta[0] = 0
    weigths -= delta
