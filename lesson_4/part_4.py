import numpy as np


def nn(input, weights):
    return input.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


input = np.array([10, 30])
weigths = np.array([[0.2, 0.3], [0.7, 0.4]]).T

true_prediction = np.array([50, 120])
learning_rate = 0.0001

for i in range(50):
    prediction = nn(input, weigths)
    error = get_error(true_prediction, prediction)
    print("Prediction: %s, Weights: %s, Error: %s" % (prediction, weigths, error))
    delta = (prediction - true_prediction) * input * learning_rate
    weigths -= delta
