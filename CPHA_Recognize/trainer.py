import  os
import  numpy as np
import  torch
import torch.nn.functional as F
import  torch.utils.data as data
import  torch.nn as nn
from  myData import dataset
img_path = "pic"
BATCH_SIZE=64
EPOCH =100
save_path = r"model/seq2seq.pkl"


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #第一层网络:用以接收数据进行形状变换
        #这里应该是一个普通的神经网络，为了得到格式化的输入输出。
        self.fcl = nn.Sequential(
            #将一张图片纵着切割
            #每次接受180的数据，然后用128个神经元去接收它
            nn.Linear(180,128),#[batch_size*120,128]
            #进行归一化
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()

        )
        #第二层网络（叠加的RNN）
        self.lstm=nn.LSTM(input_size=128, #第一层输出的结果，输入的数据
                          hidden_size=128,#隐藏层的大小，用128个细胞来处理128个数据
                          num_layers=2,
                          batch_first=True
                          )
    def forward(self,x):
        #现在x为NSV形式
        #N H W C[N 60 120 3]
        #180为步长，图片长度为120*60
        x = x.view(-1,180,240).permute(0,2,1)
        #转为NV
        # #[N*120 180] NSV->NV
        x = x.contiguous().view(-1,180)
        fc1 = self.fcl(x)

        #NV->NSV
        fc1 = fc1.view(-1,240,128);
        lstm,(hn,hc) = self.lstm(fc1)
        #只需要最后一次的输出即可
        out = lstm[:,-1,:]
        return out;
class Decodeer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size = 128,
            hidden_size=128,
            num_layers=2,
            batch_first=True#单向递归
        )
        self.out = nn.Linear(128,10)
    def forward(self,x):
        x = x.view(-1,1,128) #[N,1,128]  NV->NSV
        #将中间1一个维度填充成4个维度
        x  = x.expand(-1,4,128)     #[N 4 128]
        lstm,(h_n,h_c)=self.lstm(x)        #[N,4,128]
        y1 = lstm.contiguous().view(-1,128)#[N*4,128]
        out = self.out(y1)                 #[n*4,10]
        out = out.view(-1,4,10)           #[n,4,10]
        output = F.softmax(out,dim=2)      #输出每个的概率
        return out,output

class SEQ2SEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.endcoder = Encoder()
        self.deecoder = Decodeer()
    def forward(self,x):
        encoder_out = self.endcoder(x)
        out_put=self.deecoder(encoder_out)
        return  out_put
if __name__ == '__main__':
    seq2seq = SEQ2SEQ()

    opt = torch.optim.Adam([{'params':seq2seq.endcoder.parameters()}
                             ,{'params':seq2seq.deecoder.parameters()}])
    loss_func = nn.MSELoss()
    if os.path.exists(save_path):
        #保存权重
        seq2seq.load_state_dict(torch.load(save_path))
    train_data = dataset("pic")
    train_loader = data.DataLoader(dataset=train_data
                                   ,batch_size=BATCH_SIZE,
                                   shuffle=True,drop_last=True)
    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
            batch_x = x
            batch_y = y;
            decorder = seq2seq(batch_x)
            out,output = decorder[0],decorder[1]

            loss  = loss_func(out,batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%5==0:
                test_y = torch.argmax(y,2).data.numpy()
                pred =  torch.argmax(output,2).data.numpy()
                acc= np.mean(np.sum(pred==test_y,axis=1)==4)
                print("epoch",epoch,"i:",i,"loss:",loss.item())
                print("acc:",acc)
                print("test_y",test_y[0])
                print("pred", pred[0])

        torch.save(seq2seq.state_dict(),save_path)