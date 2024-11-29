import numpy as np


def nn(input: list[int], weights: list[int]) -> int:
    return input.dot(weights)


def get_error(true_prediction, prediction) -> int:
    return (true_prediction - prediction) ** 2


prediction = nn(np.array([150, 40]), [0.2, 0.3])

true_prediction = 50

print("get_error", get_error(true_prediction=true_prediction, prediction=prediction))
