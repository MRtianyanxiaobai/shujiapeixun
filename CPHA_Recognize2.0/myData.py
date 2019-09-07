# *_*coding:utf-8 *_*
import os
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

data_transforms = transforms.Compose([
    #将data转为张量以及归一化操作
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
class myDataset(data.Dataset):
    def __init__(self,rootpath,mapdic):
        self.transforms = data_transforms
        self.list=[]
        for filename in os.listdir(rootpath):
            x = os.path.join(rootpath,filename)

            #ys = mapdic[str(filename.split(".")[0])]
            ys = str(filename.split(".")[0])


            y = self.one_hot(ys)

            self.list.append([x, np.array(y)])

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path, label = self.list[index]
        img = Image.open(img_path)

        img = self.transforms(img)

        label = torch.from_numpy(label)

        return img, label
    def one_hot(self,x):
        z = np.zeros(shape=[4, 10])
        for i in range(4):
            index = int(x[i])
            z[i][index] += 1
        return z

    # def one_hot(self, x):
    #     z = np.zeros(shape=[5, 36])
    #     for i in range(5):
    #
    #         if x[i]<='9' and x[i]>='0':
    #             index = int(x[i])
    #             z[i][index] += 1
    #         elif x[i]<='z' and x[i]>='a':
    #             index = int(ord(x[i])-ord('a'))+10
    #             z[i][index] += 1
    #         else:
    #             index = int(ord(x[i])-ord('A'))+10
    #             z[i][index] += 1
    #
    #     return z
    def reonhot(self,x):
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
filedir_path = r"F:\verifycodes\data-2\train"
map_path = r"F:\verifycodes\data-2\mappings.txt"
dic = getdic(map_path)
mydata = myDataset(filedir_path, dic)

    # def one_hot(self,x):
    #     z = np.zeros(shape=[4, 10])
    #     for i in range(4):
    #         index = int(x[i])
    #         z[i][index] += 1
    #     return z
