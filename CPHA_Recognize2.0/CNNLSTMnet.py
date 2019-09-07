# *_*coding:utf-8 *_*
# *_*coding:utf-8 *_*
import  os
import  numpy as np
import  torch
import torch.nn.functional as F
import  torch.utils.data as data
import  torch.nn as nn
import torch
import  torchvision
import myData
BATCH_SIZE=100
EPOCH =1000
# filedir_path = r"F:\verifycodes\data-2\train"
filedir_path = r"E:\ShuJiaPeiXun\710\pic"
map_path = r"F:\verifycodes\data-2\mappings.txt"
def one_hot(x):
    z = np.zeros(shape=[5, 62])
    for i in range(5):

        if x[i] <= '9' and x[i] >= '0':
            index = int(x[i])
            z[i][index] += 1
        elif x[i] <= 'z' and x[i] >= 'a':
            index = int(ord(x[i]) - ord('a')) + 10
            z[i][index] += 1
        else:
            index = int(ord(x[i]) - ord('A')) + 36
            z[i][index] += 1

    return z
def reonhot(x):
    test_y = np.argmax(x, 1)
    list=[]
    for i in range(5):
        if test_y[i]<=9 and test_y[i]>=0:
            list.append(test_y[i])
        else:
            list.append(chr(test_y[i]-10+ord('A')))
    return list;
def getdic(map_path):
    dic = {}
    keys = []  # 用来存储读取的顺序
    fr = open(map_path, 'r')
    for line in fr:
        v = line.strip().split(',')
        dic[v[0]] = v[1]
        keys.append(v[0])
    fr.close()
    return dic
class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = self.build(3, 32,[2, 2, 2, 2, 2],[32, 64, 128, 256, 512])
        self.out = nn.Linear(in_features=512,out_features=10)

    def forward(self,x):
        vggout = self.blocks(x)
        vggout = vggout.contiguous().view(-1,1,512)
        vggout = vggout.expand(-1,4,512)
        vggout = vggout.contiguous().view(-1,512)
        out = self.out(vggout)
        out = out.contiguous().view(-1,4,10)
        output = F.softmax(out,dim=2)
        return out,output

    def build(self,inchas,outchas,conv_nums, filters):
        """
        Build CNN like vgg
        use GlobalAveragePooling to reduce parameters
        :param width: Integer. The width of the image
        :param height: Integer. The height of the image
        :param depth: Integer. The depth of the image
        :param conv_nums: 1d array. Conv nums  of each stage
        :param filters: 1d array. filters nums of each stage
        :param l2_rate: Float. L2 regularizer rate of the conv layer.
        """
        net = [nn.Conv2d(inchas,outchas,kernel_size=3,padding=1),nn.ReLU(True)]

        for block,(conv_num,filter) in enumerate(zip(conv_nums,filters)):
            for stage in  range(conv_num):
                net.append(nn.Conv2d(outchas,filter,kernel_size=3,padding=1))
                net.append(nn.ReLU(True))
                net.append(nn.BatchNorm2d(filter))
                outchas = filter
            if block == len(conv_nums)-1:
                net.append(nn.AdaptiveAvgPool2d((1,1)))
                net.append(nn.Dropout(p=0.5))
            else:
                net.append(nn.MaxPool2d(2,2))
                net.append(nn.Dropout(p=0.25))
        return nn.Sequential(*net);

# block_demo = build(3, 32,[2, 2, 2, 2, 2],[32, 64, 128, 256, 512])
# from torch.autograd import Variable
# # print(block_demo)
# input_demo = Variable(torch.zeros(1,3, 40, 100))
# net = VGGNet()
# (out,output) = net(input_demo)
# print(out.size())
# print(out.size())
# output_demo = block_demo(input_demo)
# print(output_demo.shape)

if __name__ == '__main__':
    seq2seq = VGGNet()
    opt = torch.optim.Adam(seq2seq.parameters())
  #  opt = torch.optim.SGD(seq2seq.parameters(),lr=0.01)
    #seq2seq.load_state_dict(torch.load("1.pth"))
    loss_func = nn.MSELoss()

    dic = getdic(map_path)
    mydata = myData.myDataset(filedir_path, dic)
    train_loader = data.DataLoader(dataset=mydata
                                   , batch_size=BATCH_SIZE,
                                   shuffle=True, drop_last=True)

    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
          #  print(x.size())
            decoder = seq2seq(x)
            out, output = decoder[0], decoder[1]
            loss =  loss_func(out, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 5 == 0:
                test_y = torch.argmax(y, 2).data.numpy()
                pred_y = torch.argmax(output, 2).data.numpy()
                acc = np.mean(np.sum(pred_y == test_y, axis=1) == 4)
                print("epoch:",epoch)
                print(loss)
                print(acc)





        torch.save(seq2seq.state_dict(),"2vgg.pth")