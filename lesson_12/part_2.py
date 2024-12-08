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
                    self.creators[0].backward(self.grad.__neg__())
                elif self.operation_on_creation == "-":
                    self.creators[0].backward(self.grad)
                    self.creators[1].backward(self.grad.__neg__())

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


t_1 = Tensor([1, 2, 3], autograd=True)
t_2 = Tensor([4, 5, 6], autograd=True)
t_3 = Tensor([7, 8, 9], autograd=True)
a_add_1 = t_1 - t_2
a_add_2 = t_2 - t_3
a_add_3 = a_add_1 - a_add_2
a_add_3.backward(Tensor([4, 5, 3]))

print(t_3.grad)
