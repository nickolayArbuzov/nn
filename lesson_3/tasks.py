import numpy as np


# 1:
def nn(input: list[int], weights: list[int]) -> int:
    return input.dot(weights)


def get_error(true_prediction, prediction) -> int:
    return (true_prediction - prediction) ** 2


prediction = nn(np.array([150, 40]), [0.2, 0.4999])

true_prediction = 50

print("# 1: ", get_error(true_prediction=true_prediction, prediction=prediction))

# 2:
weight = 10
error = 10**10
while error > 0.001:
    prediction = nn(np.array([150, 40]), [0.2, weight])
    error = get_error(true_prediction=true_prediction, prediction=prediction)
    delta = (prediction - true_prediction) * 0.01
    weight -= delta

print(
    "# 2: ",
    "Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error),
)


# 3:
def nn_2(input: int, weights: int) -> int:
    return input * weights


input = 0.9
weight = 0.2

prediction = nn_2(input, weight)

true_prediction = 0.8

for i in range(15):
    prediction = nn_2(input, weight)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error))
    delta = (prediction - true_prediction) * input
    weight -= delta
