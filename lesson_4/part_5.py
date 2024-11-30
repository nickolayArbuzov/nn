import numpy as np


def nn(input, weights):
    return input.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


input = np.array([[150, 40], [170, 80], [160, 90]])
weigths = np.array([0.2, 0.3]).T

true_predictions = np.array([50, 120, 140])
learning_rate = 0.00004

for i in range(500):
    error = 0
    for j in range(len(input)):
        prediction = nn(input[j], weigths)
        error += get_error(true_predictions[j], prediction)
        print("Prediction: %s, Weights: %s" % (prediction, weigths))
        delta = (prediction - true_predictions[j]) * input[j] * learning_rate
        weigths -= delta
    print("Error: %.10f", (error))
    print("-------------------------")
