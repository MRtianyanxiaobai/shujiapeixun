# *_*coding:utf-8 *_*
import  os
import  numpy as np
import  torch
import torch.nn.functional as F
import  torch.utils.data as data
import  torch.nn as nn
import torch
BATCH_SIZE=100
EPOCH =100
filedir_path = r"F:\verifycodes\data-2\train"
map_path = r"F:\verifycodes\data-2\mappings.txt"
dic = {}
keys = [] #用来存储读取的顺序
fr=open(map_path,'r')
for line in fr:
    v = line.strip().split(',')
    dic[v[0]] = v[1]
    keys.append(v[0])
fr.close()
