import numpy as np


def nn(input, weights):
    return input.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


input = np.array(
    [
        [150, 40],
        [170, 80],
        [160, 90],
        [180, 90],
        [160, 45],
        [170, 57],
        [175, 75],
        [165, 50],
    ]
)
weigths = np.array([0.2, 0.3])

true_predictions = np.array([0, 100, 100, 100, 0, 0, 100, 0])
learning_rate = 0.00003

for i in range(500):
    error = 0
    delta = 0
    for j in range(len(input)):
        prediction = nn(input[j], weigths)
        error += get_error(true_predictions[j], prediction)
        print(
            "Prediction: %s, True_prediction: %s, Weights: %s"
            % (prediction, true_predictions[j], weigths)
        )
        delta += (prediction - true_predictions[j]) * input[j] * learning_rate
    weigths -= delta / len(input)
    print("Error: %.10f", (error))
    print("-------------------------")

print(nn(np.array([169, 52]), weigths))
print(nn(np.array([169, 72]), weigths))
print(nn(np.array([179, 52]), weigths))
print(nn(np.array([179, 72]), weigths))
