#encoding=utf8
import  torch
import  torch.nn as nn
class Net(nn.Module):
    #构造方法，初始化网络结构
    def __init__(self):
        super().__init__()
        #相当于流水线，帮助我们组装网络体系
        self.layers=nn.Sequential(
            #第一层网络，输入层
            #第一个参数输入的个数
            # 第二个参数是神经元的个数（超参数）--数据的个数
            nn.Linear(28*28,512),
            nn.ReLU(),#激活函数、
            nn.Linear(512,256),#第二层
            nn.ReLU(),
            nn.Linear(256,128),#第三层
            nn.ReLU(),
            #输出层的神经元和结果有关
            nn.Linear(128,10),#输出层

            #利用多分类激活函数
            nn.Softmax()

        )
    #前项计算,相当于线程中的run方法
    #使用网络
    def forward(self,x):
        output = self.layers(x)
        return output;
