from test06 import dataset
from torch.utils import data
import torch
import  torchvision
from  PIL import  Image,ImageDraw
from  myNet import  Net
from torch.utils import data
import  torch.nn as nn
import numpy as np
if __name__ == '__main__':
    data_set = dataset("test_pic2")
    # shuffle 打乱数据之间的连续性
    # batch_size 一次所取的数据 这里一次取100张图片
    # 使用数据加载器，从dataset中打乱取数据
    train_data=data.DataLoader(dataset=data_set,batch_size=100
                               ,shuffle=True)

    #net = Net()
    net = torch.load("1.pth")
    # 计算损失函数
    loss_func = nn.MSELoss()
    # 利用梯度下降优化所有的参数 lr:步长
    # optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    # optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(1000):
        # 用枚举的形式获得下标，index
        # x:数据 y：标签
        for i, (x, y) in enumerate(train_data):
            # 将x做形状变化，2828化成764*（一维）

            x = x.view(-1, 300*300*3)  # 其中-1代表剩余的部分
            out = net(x)
                       # 将标签变成one-hot形式
            x = x.view(-1, 300,300, 3)
            out=out.detach().numpy()*300
            y =y.detach().numpy()*300
            print(out[0])
            img_data = np.array((x[0]+0.5)*255,dtype=np.int8)
            img = Image.fromarray(img_data,"RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle(out[0],outline="red",width=2)
            draw.rectangle(y[0], outline="green", width=2)
            print(y[0])
            img.show()

            # # 传入损失
            # loss = loss_func(out, y)
            #
            # # 梯度下降步骤
            # optimizer.zero_grad()  # 清空梯度
            # loss.backward()  # 自动求导
            # optimizer.step()  # 更新梯度

        #     if i % 10 == 0:
        #         # out_put = torch.argmax(out, dim=1)
        #         print(loss)
        #
        #         # print("target:",target)
        #         # print("out:",out_put)
        #         acc = np.mean(np.sum(out.detach().numpy()==y.numpy(),axis=1)==4);
        #         print(acc)
        # torch.save(net, "1.pth")