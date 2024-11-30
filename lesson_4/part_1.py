def nn(input: int, weights: int) -> int:
    return input * weights


def get_error(true_prediction, prediction) -> int:
    return (true_prediction - prediction) ** 2


input = 70
weight = 0.1

prediction = nn(input, weight)

true_prediction = 40
learning_rate = 0.0004

for i in range(100):
    prediction = nn(input, weight)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error))
    delta = (prediction - true_prediction) * input * learning_rate
    weight -= delta
