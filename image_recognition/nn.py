import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

root = "./images"
batch_size = 10

transformations = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = CIFAR10(train=True, transform=transformations, root=root, download=True)
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = CIFAR10(train=False, transform=transformations, root=root, download=True)
test_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

# output_size_edge = (input_size_edge - kernel_size + 2 * padding) / stride + 1


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        # (32 - 5 + 2 * 1) / 1 + 1 = 30
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(12)
        # (30 - 5 + 2 * 1) / 1 + 1 = 28
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(12)
        # 28 / 2 = 14
        self.pool = nn.MaxPool2d(2, 2)
        # (14 - 5 + 2 * 1) / 1 + 1 = 12
        self.conv3 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(24)
        # (12 - 5 + 2 * 1) / 1 + 1 = 10
        self.conv4 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(24)
        self.fc = nn.Linear(24 * 10 * 10, 10)

    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = out.view(-1, 24 * 10 * 10)
        return self.fc(out)


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
model = ImageModel()


def test_accuracy():
    model.eval()
    accuracy = 0
    total = 0
    for test_data in test_data_loader:
        images, labels = test_data
        output = model(images)
        predicted = torch.max(output.data, 1)[1]
        accuracy += (predicted == labels).sum().item()
        total += labels.size(0)

    return 100 * accuracy / total


loss = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

num_epochs = 3
best_accaracy = 0.0
model_save_path = "./best.pth"

model.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data_loader, 0):
        optimizer.zero_grad()
        output = model(images)
        error = loss(output, labels)
        error.backward()
        optimizer.step()
    accuracy = test_accuracy()
    print("epoch: ", epoch, " accuracy: ", accuracy)
    if accuracy > best_accaracy:
        best_accaracy = accuracy
        torch.save(model.state_dict(), model_save_path)


def print_labels(title, labels):
    print(title, end=" ")
    for i in range(batch_size):
        print(classes[labels[i]], end=" ")
    print()


load_model = ImageModel()
load_model.load_state_dict(torch.load(model_save_path))
images, true_labels = next(iter(test_data_loader))
print_labels("labels: ", true_labels)

output = load_model(images)
predicted_labels = torch.max(output, 1)[1]
print_labels("predicted labels: ", predicted_labels)

images = torchvision.utils.make_grid(images)
images = images / 2 + 0.5
plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
plt.show()
