from cifar10_pyTorch import *
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import argparse


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.0')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save', type=bool, default=True,
                        help='save model (default: True)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Continue training from a previous checkpoint')
    args = parser.parse_args()

    # Training settings
    train_loader, test_loader, categories = init(args.batch_size)
    visualize(train_loader, categories)
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    summary(net, (3, 32, 32))
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Resume training?
    if args.continue_from is not None:
        net.load_state_dict(torch.load(args.continue_from))

    # Train
    train(num_epochs=args.epochs, loader=train_loader, evaluation_loader=test_loader,
          device=device, optimizer=optimizer, criterion=criterion, net=net)

    # Final evaluations
    train_acc = inference(loader=train_loader, device=device, net=net)
    test_acc = inference(loader=test_loader, device=device, net=net)

    # Save parameters
    if args.save:
        torch.save(net.state_dict(), str(time.time()) + '.pth')
