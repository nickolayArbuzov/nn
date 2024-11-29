import numpy as np


def nn(input: int, weights: int) -> int:
    return input * weights


def get_error(true_prediction, prediction) -> int:
    return (true_prediction - prediction) ** 2


input = 0.9
weight = 0.1

prediction = nn(input, weight)

true_prediction = 0.2

for i in range(10):
    prediction = nn(input, weight)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error))
    delta = (prediction - true_prediction) * input
    weight -= delta
