import numpy as np


class Tensor(object):
    def __init__(self, data, creators=None, operation_on_creation=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None

    def __add__(self, other):
        return Tensor(self.data + other.data, [self, other], "+")

    def backward(self, grad):
        self.grad = grad
        if self.operation_on_creation == "+":
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)

    def __str__(self):
        return str(self.data.__str__())


t_1 = Tensor([3, 5, 10])
t_2 = Tensor([4, 1, 9])
t_3 = Tensor([3, 5, 10])
t_4 = Tensor([4, 1, 9])
a_add_1 = t_1 + t_2
a_add_2 = t_3 + t_4
a_add_3 = a_add_1 + a_add_2
a_add_3.backward(Tensor([4, 5, 3]))

print(t_1.grad)
