import torch
import  torchvision
from  PIL import  Image
from  myNet import  Net
from torch.utils import data
import  torch.nn as nn
import numpy as np
train_data = torchvision.datasets.MNIST(
    root = "MNIST_data/",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_data = torchvision.datasets.MNIST(
    root = "MNIST_data/",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

net = torch.load("1.pth")
test = data.DataLoader(dataset=test_data,
                        batch_size=10000, shuffle=True)

# 计算损失函数
loss_func = nn.MSELoss()
for i, (x, y) in enumerate(test):
    # 将x做形状变化，28*28化成764（一维）
    x = x.view(-1, 28 * 28)  # 其中-1代表剩余的部分
    out = net(x)
    # 将标签变成one-hot形式
    y = y.long()
    target = torch.zeros(y.size()[0], 10).scatter_(1, y.view(-1, 1), 1)
    # 传入损失
    loss = loss_func(out, target)
    out_put = torch.argmax(out, dim=1)
    print(loss)
    acc = np.mean(np.array(out_put == y, dtype=np.float32))
    print(acc)




