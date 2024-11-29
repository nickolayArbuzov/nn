def nn(input: list[int], weights: list[list[int]]) -> list[int]:
    prediction = []
    for i in range(len(weights)):
        ws = 0
        for j in range(len(input)):
            ws += input[j] * weights[i][j]
        prediction.append(ws)
    return prediction


out_1 = nn(input=[30, 70], weights=[[0.4, 0.3], [0.1, 0.4]])

print(out_1)
