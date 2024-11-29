def nn(input: int, weights: list[int]) -> list[int]:
    prediction = []
    for i in range(len(weights)):
        prediction.append(input * weights[i])
    return prediction


out_1 = nn(input=40, weights=[0.2, 0.3])
out_2 = nn(input=120, weights=[0.4, 0.3])

print(out_1)
print(out_2)
