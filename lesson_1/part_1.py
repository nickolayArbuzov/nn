def nn(input: int, weight: int) -> int:
    prediction = input * weight
    return prediction


out_1 = nn(input=170, weight=0.2)
out_2 = nn(input=100, weight=0.2)

print(out_1)
print(out_2)
