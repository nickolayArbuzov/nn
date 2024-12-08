import numpy as np


class Tensor(object):
    def __init__(
        self, data, creators=None, operation_on_creation=None, autograd=False, id=None
    ):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if id is None:
            id = np.random.randint(0, 9**9)
        self.id = id

        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
            if self.creators is not None and (
                self.check_grads_from_children() or grad_origin is None
            ):
                if self.operation_on_creation == "+":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                elif self.operation_on_creation == "-1":
                    self.creators[0].backward(self.grad.__neg__(), self)
                elif self.operation_on_creation == "-":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)
                elif self.operation_on_creation == "*":
                    self.creators[0].backward(self.grad * self.creators[1], self)
                    self.creators[1].backward(
                        self.grad * self.creators[0].__neg__(), self
                    )
                elif "sum" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(
                        self.grad.expand(axis, self.creators[0].data.shape[axis]), self
                    )
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.sum(axis), self)
                elif self.operation_on_creation == "dot":
                    temp = self.grad.dot(self.creators[1].transpose())
                    self.creators[0].backward(temp, self)
                    temp = self.grad.dot(self.creators[0].transpose())
                    self.creators[1].backward(temp, self)
                elif self.operation_on_creation == "transpose":
                    return self.creators[0].backward(self.grad.transpose(), self)
                elif self.operation_on_creation == "sigmoid":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (temp - self)), self)
                elif self.operation_on_creation == "tanh":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (temp - self * self), self)
                elif self.operation_on_creation == "softmax":
                    self.creators[0].backward(Tensor(self.grad.data))

    def check_grads_from_children(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
            return True

    def __str__(self):
        return str(self.data.__str__())

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, [self], "-1", True)
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self, other], "-", True)
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))

    def expand(self, axis, count_copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        expand_shape = list(self.data.shape) + [count_copies]
        expand_data = self.data.repeat(count_copies).reshape(expand_shape)
        expand_data = expand_data.transpose(transpose)
        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)
        return Tensor(expand_data)

    def dot(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        return Tensor(self.data.transpose())

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)), [self], "sigmoid", True)
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh", True)
        return Tensor(np.tanh(self.data))

    def softmax(self):
        exp = np.exp(self.data)
        exp = exp / np.sum(exp, axis=1, keepdims=True)
        if self.autograd:
            return Tensor(exp, [self], "softmax", True)
        return Tensor(exp)

    def __repr__(self):
        return str(self.data.__repr__())


class SGD(object):
    def __init__(self, weights, learning_rate=0.01):
        self.weights = weights
        self.learning_rate = learning_rate

    def step(self):
        for weight in self.weights:
            weight.data -= self.learning_rate * weight.grad.data
            weight.grad.data *= 0


class Layer(object):
    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, input_count, output_count):
        super().__init__()
        weight = np.random.randn(input_count, output_count) * np.sqrt(2 / input_count)
        self.weight = Tensor(weight, autograd=True)
        self.bias = Tensor(np.zeros(output_count), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.dot(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params


class Sigmoid(Layer):
    def forward(self, input):
        return input.sigmoid()


class Tanh(Layer):
    def forward(self, input):
        return input.tanh()


class Softmax(Layer):
    def forward(self, input):
        return input.softmax()


class MSELoss(Layer):
    def forward(self, prediction, true_prediction):
        return ((prediction - true_prediction) * (prediction - true_prediction)).sum(0)


np.random.seed(0)

x = Tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
    ],
    autograd=True,
)
y = Tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    autograd=True,
)

model = Sequential([Linear(4, 15), Sigmoid(), Linear(15, 10), Softmax()])
sgd = SGD(model.get_parameters(), 0.01)
loss = MSELoss()
epochs = 100000

for epoch in range(epochs):
    predictions = model.forward(x)
    error = loss.forward(prediction=predictions, true_prediction=y)
    error.backward(Tensor(np.ones_like(error.data)))
    sgd.step()

    if epoch % 1000 == 0:
        print("error", error)


def predict(input):
    output_layer = model.forward(input)
    return np.argmax(output_layer.data)


inputs = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
]

for input in inputs:
    print(f"digit - {input}: ", predict(Tensor([input])))
