import  numpy as np
#实现one-hot编码
import  torch
y= torch.Tensor([8,5,3])
index = y.view(-1,1).long()#转成列向量，
        # 并将类型转换为scatter能够接受的long类型
#scatter：向某一个维度填充数据(向列维度中的指定index添加1)
z = torch.zeros(y.size()[0],10).scatter_(1,y.view(-1,1).long(),1)
#将维度转回来
torch.argmax(out,dim=1)


#print(ju)