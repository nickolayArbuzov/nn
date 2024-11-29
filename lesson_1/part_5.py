def nn(input: list[int], weights: list[list[list[int]]]) -> list[int]:
    prediction_hidden = [0] * len(weights[0])

    for i in range(len(weights[0])):
        ws = 0
        for j in range(len(input)):
            ws += input[j] * weights[0][i][j]
        prediction_hidden[i] = ws

    prediction_out = [0] * len(weights[1])

    for i in range(len(weights[1])):
        ws = 0
        for j in range(len(input)):
            ws += prediction_hidden[j] * weights[1][i][j]
        prediction_out[i] = ws

    return prediction_out


input = [50, 165]
weights_hidden_1 = [0.2, 0.1]
weights_hidden_2 = [0.3, 0.1]
weights_out_1 = [0.4, 0.2]
weights_out_2 = [0.5, 0.3]
weights_hidden = [weights_hidden_1, weights_hidden_2]
weights_out = [weights_out_1, weights_out_2]
weights = [weights_hidden, weights_out]

out_1 = nn(input=input, weights=weights)

print(out_1)
