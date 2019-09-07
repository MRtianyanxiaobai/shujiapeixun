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
# print(torch.__version__)
# img = train_data.data[0].numpy()
# img = Image.fromarray(img,"L")
# img.show()
# tlabe = train_data.targets[0].numpy();
# print(tlabe)

if __name__ == '__main__':
    #shuffle 打乱数据之间的连续性
    #batch_size 一次所取的数据 这里一次取100张图片
    #使用数据加载器，从dataset中打乱取数据
    train = data.DataLoader(dataset=train_data,
                                batch_size=100,shuffle=True)
    net = Net()
    #计算损失函数
    loss_func = nn.MSELoss()
    #利用梯度下降优化所有的参数 lr:步长
   # optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in  range(10):
        #用枚举的形式获得下标，index
        #x:数据 y：标签
        for i, (x, y) in enumerate(train):
            # 将x做形状变化，2828化成764*（一维）
            x = x.view(-1, 28 * 28)  # 其中-1代表剩余的部分
            out = net(x)
            #将标签变成one-hot形式
            y = y.long()
            target = torch.zeros(y.size()[0], 10).scatter_(1, y.view(-1, 1), 1)
            # print(target.size())
            # 传入损失
            loss = loss_func(out, target)

            # 梯度下降步骤
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 自动求导
            optimizer.step()  # 更新梯度

            if i % 10 == 0:
                out_put = torch.argmax(out, dim=1)
                print(loss)
                # print("target:",target)
                # print("out:",out_put)
                acc = np.mean(np.array(out_put == y, dtype=np.float32))
                print(acc)
        torch.save(net,"1.pth")
