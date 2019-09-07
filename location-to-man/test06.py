import os
import torch
from torch.utils import data
from PIL import  Image,ImageDraw
import  numpy as np

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
        label = self.dataset[index].split(".")[1:5]
        #想转换成int，然后转成pytorch认识的tensor
        label = torch.Tensor(np.array(label,dtype=np.float32));
        img_path=os.path.join(self.path,self.dataset[index])
        img = Image.open(img_path).convert("RGB")
        img_data =torch.Tensor( np.array(img)/255-0.5)
        return img_data,label/300

# if __name__ == '__main__':
#     mydata = dataset("test_pic2")
#     x = mydata[112][0].numpy()
#     y = mydata[112][1].numpy()
#     #print(x)
#    # print(y)
#     img_data = np.array((x+0.5)*255,dtype=np.int8)
#     img = Image.fromarray(img_data,"RGB")
#
#     draw = ImageDraw.Draw(img)
#     draw.rectangle(y,outline="red",width=2)
#     img.show()