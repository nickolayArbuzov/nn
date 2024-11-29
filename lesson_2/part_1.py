import numpy as np


def nn(input: list[int], weights: list[list[list[int]]]) -> list[int]:
    prediction_hidden = input.dot(weights[0])
    prediction_out = prediction_hidden.dot(weights[1])

    return prediction_out


input = np.array([50, 165])
weights_hidden_1 = [0.2, 0.1]
weights_hidden_2 = [0.3, 0.1]
weights_out_1 = [0.4, 0.2]
weights_out_2 = [0.5, 0.3]
weights_hidden = np.array([weights_hidden_1, weights_hidden_2]).T
weights_out = np.array([weights_out_1, weights_out_2]).T
weights = [weights_hidden, weights_out]

out_1 = nn(input=input, weights=weights)

print(out_1)
