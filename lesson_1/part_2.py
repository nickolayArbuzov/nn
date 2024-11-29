def scalar_multiply(input: list[int], weights: list[int]) -> int:
    prediction = 0
    for i in range(len(input)):
        prediction += input[i] * weights[i]
    return prediction


out_1 = scalar_multiply(input=[150, 40], weights=[0.2, 0.3])
out_2 = scalar_multiply(input=[120, 30], weights=[0.2, 0.3])

print(out_1)
print(out_2)
