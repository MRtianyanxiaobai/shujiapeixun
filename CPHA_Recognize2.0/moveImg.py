# *_*coding:utf-8 *_*
import os, sys
from PIL import Image
frompath = r"F:\verifycodes\data-2\train"
testpath = r"F:\verifycodes\data-2\test"
for index in range(8000,10001):
    img = Image.open(frompath+"\\"+str(index)+".jpg")
    img.save(testpath+"\\"+str(index)+".jpg")
    os.remove(frompath+"\\"+str(index)+".jpg")