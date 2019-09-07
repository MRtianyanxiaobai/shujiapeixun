# *_*coding:utf-8 *_*
import torch
import  torchvision
from  PIL import  Image
from torch.utils import data
import  torch.nn as nn
import numpy as np
import  os
import myData
filedir_path = r"F:\verifycodes\data-2\train"
map_path = r"F:\verifycodes\data-2\mappings.txt"



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
dic = getdic(map_path)
mydata = myData.myDataset(filedir_path,dic)
train_data = data.DataLoader(dataset=mydata,shuffle=True,batch_size=1000)
for i,(x,y) in enumerate(train_data):
    print(i)
    print(x.size())
    print(y.size())