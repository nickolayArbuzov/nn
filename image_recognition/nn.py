from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

root = "./images"
batch_size = 10

transformations = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = CIFAR10(train=True, transform=transformations, root=root, download=True)
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = CIFAR10(train=False, transform=transformations, root=root, download=True)
test_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
