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
filedir_path = r"F:\verifycodes\data-2\train"
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
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180,128),#[batch_size*120,128]
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        # self.lstm2 = nn.LSTM
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
    def forward(self, x):           #[N,3,60,120]
        x = x.view(-1,180,200).permute(0,2,1) #[N,120,180]  NCHW-->NSV
        x = x.contiguous().view(-1,180)  #[N*120,180]   NSV-->NV
        fc1 = self.fc1(x)               #[N*120,128]  NV
        fc1 = fc1.view(-1,200,128)      #[N,120,128] NV-->NSV
        lstm,(h_n,h_c) = self.lstm(fc1)
        out = lstm[:,-1,:]   #NSV需要的是最后一个S的V
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(128,36)
    def forward(self, x):
        x = x.view(-1,1,128)      #[N,1,128]  NV--》NSV
        x = x.expand(-1,5,128)      #[N,4,128]
        lstm,(h_n,h_c) = self.lstm(x)   #[N,4,128]
        y1 = lstm.contiguous().view(-1,128)  #[N*4,128]
        out = self.out(y1)                   #[N*4,10]
        out = out.view(-1,5,36)                 #[N,4,10]
        output = F.softmax(out,dim=2)

        return out,output

class SEQ2SEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        encoder_out = self.encoder(x)
        out_put = self.decoder(encoder_out)
        return out_put
if __name__ == '__main__':
    seq2seq = SEQ2SEQ()
    opt = torch.optim.Adam([{'params': seq2seq.encoder.parameters()},
                            {"params": seq2seq.decoder.parameters()}])
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
                acc = np.mean(np.sum(pred_y == test_y, axis=1) == 5)
                print(loss)
                print(acc)





        torch.save(seq2seq.state_dict(),"1.pth")