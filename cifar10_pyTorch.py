import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchsummary import summary


vgg_B = [64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512]


# Network architecture(like vgg-B)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convs = generate_layers_vgg(vgg_B)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcs = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
            nn.ReLU(inplace=True)
            )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.convs(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x


def generate_layers_vgg(config):
    layers = []
    in_channels = 3
    for out_channels in config:
        # max pool
        if out_channels == -1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # convolution
        else:
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

    return nn.Sequential(*layers)


# Draw images
def show(images):
    images = images / 2 + 0.5  # denormalize
    np_images = images.numpy()
    print(np_images.shape)
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    plt.show()


# Show random images
def visualize(loader):
    temp = iter(loader)
    images, labels = temp.next()
    show(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(labels.size(0))))


# Load data
def init(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    print(type(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


# Train data
def train(num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        time_now = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        print('Epoch time: %.2fs' % (time.time() - time_now))


# Test
def inference(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test acc: %f' % (100 * correct / total))


if __name__ == "__main__":
    train_loader, test_loader, classes = init(32)
    visualize(train_loader)
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    summary(net, (3, 32, 32))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    #train(2)
    inference(test_loader)
    torch.save(net, str(time.time()))
