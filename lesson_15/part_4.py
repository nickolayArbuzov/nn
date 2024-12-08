from torch import nn
import torch


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


torch.manual_seed(0)
input_size = 3
hidden_size = 4
output_size = 3
learning_rate = 0.1
num_epochs = 1000

x = torch.randn(100, input_size)
y = torch.randint(0, output_size, (100,))

model = SimpleModel(input_size, hidden_size, output_size)
"""loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(x)
    error = loss(out, y)
    error.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(epoch + 1, "Потери: ", error.item())

print(model.state_dict())
torch.save(model.state_dict(), "model.pth")"""

model.load_state_dict(torch.load("model.pth"))

print(model(torch.randn(10, input_size)))
