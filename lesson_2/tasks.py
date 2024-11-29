import numpy as np

# 1:
array = np.array([1, 2, 3, 4, 5])
print("# 1: ", array)

# 2:
print("# 2: ", array * 2)

# 3:
td_array_1 = np.random.rand(3, 3)
td_array_2 = np.random.rand(3, 3)
print("# 3: ", td_array_1)
print("# 3: ", td_array_2)

# 4:
print("# 4: ", td_array_1 * td_array_2)

# 5:
od_array = np.random.randint(low=0, high=9, size=10)
print("# 5: ", od_array)

# 6:
print("# 6: ", od_array[od_array % 2 == 0])
print("# 6: ", od_array[::2])

# 7:
print("# 7: ", od_array.max())
print("# 7: ", od_array.min())
print("# 7: ", od_array.mean())
print("# 7: ", od_array.std())


# 8:
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

print("# 8: ", out_1)


# 9:
def nn(input: list[int], weights: list[list[list[int]]]) -> list[int]:
    prediction_hidden = input.dot(weights[0])
    prediction_out = prediction_hidden[0:2].dot(weights[1])

    return prediction_out


input = np.array([50, 165])
weights_hidden_1 = [0.2, 0.1]
weights_hidden_2 = [0.3, 0.1]
weights_hidden_3 = [0.6, 0.2]
weights_out_1 = [0.4, 0.2]
weights_out_2 = [0.5, 0.3]
weights_out_3 = [0.7, 0.4]
weights_hidden = np.array([weights_hidden_1, weights_hidden_2, weights_hidden_3]).T
weights_out = np.array([weights_out_1, weights_out_2, weights_out_3]).T
weights = [weights_hidden, weights_out]

out_2 = nn(input=input, weights=weights)
print("# 9: ", out_2)


# 10:
def nn(input: list[int], weights: list[list[list[int]]]) -> list[int]:
    prediction_hidden = input.dot(weights[0])
    prediction_out = prediction_hidden[0:2].dot(weights[1])

    return prediction_out


input = np.random.randint(low=0, high=200, size=2)
weights_hidden_1 = np.random.randint(low=0, high=100, size=2) / 100
weights_hidden_2 = np.random.randint(low=0, high=100, size=2) / 100
weights_hidden_3 = np.random.randint(low=0, high=100, size=2) / 100
weights_out_1 = np.random.randint(low=0, high=100, size=2) / 100
weights_out_2 = np.random.randint(low=0, high=100, size=2) / 100
weights_out_3 = np.random.randint(low=0, high=100, size=2) / 100
weights_hidden = np.array([weights_hidden_1, weights_hidden_2, weights_hidden_3]).T
weights_out = np.array([weights_out_1, weights_out_2, weights_out_3]).T
weights = [weights_hidden, weights_out]

out_3 = nn(input=input, weights=weights)

print("# 10: ", out_3)
