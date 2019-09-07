import os
import torch
from torch.utils import data
from PIL import  Image,ImageDraw
import  numpy as np
from   torchvision import  transforms

# data_transforms = transforms([  transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5],std=[0.5])
#
# ])

class dataset(data.Dataset):
    def __init__(self,path):
        self.path = path
        self.dataset = []
        #把一个列表当成元素添加进去
        self.dataset.extend(os.listdir(path))

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        #这里取出来其实是str
        label =self.dataset[index].split(".")[0]
        #想转换成int，然后转成pytorch认识的tensor
       # label = torch.Tensor(np.array(label,dtype=np.float32));
       #构造x
        img_path=os.path.join(self.path,self.dataset[index])
        img = Image.open(img_path)
        #将image进行归一化处理
        img_data =torch.Tensor( np.array(img)/255-0.5)
        labellist =np.array(list(str(label)),np.int8)
        #print(labellist)
        y = torch.Tensor(labellist);
        y=y.long()
       # print(y)
        label= torch.zeros(4, 10).scatter_(1, y.view(-1,1), 1)

        return img_data,label

if __name__ == '__main__':
    mydata = dataset("pic")

    # shuffle 打乱数据之间的连续性
    # batch_size 一次所取的数据 这里一次取100张图片
    # 使用数据加载器，从dataset中打乱取数据
    train_data = data.DataLoader(dataset=mydata, batch_size=100
                                 , shuffle=True)
    for i, (x, y) in enumerate(train_data):
        print(x)
        print(y)
        # img_data = np.array((x+0.5)*255,dtype=np.int8)
        # img = Image.fromarray(img_data,"RGB")
        # img.show()
