# cifar10
PyTorch exercise

#### With BN-VGG-16(smaller, dropout, BN):

Reached **_72%_**(test-val) without augmentations

With augmentations(padded-4 crop, horizontal flip, 15 degree rotation), batch size 64, trained with early-stopping: 

**1. SGD + momentum** 

lr=0.01&momentum=0.9 * 30 epochs *(89.1%)* + lr=0.001&momentum=0.9 * 10 epochs *(90.9%)* + lr=0.0001&momentum=0.9 * 10 epochs *(91.0%)*

**2. Adam** 

lr=0.01 * 30 epochs *(40.8%)* WTF???