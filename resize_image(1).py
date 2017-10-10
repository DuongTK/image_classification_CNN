import PIL
from PIL import Image
import os
from array import *
from random import shuffle
import re

basewidth = 32

#Names = [['./training-images','train'], ['./test-images','test']]
name = "C:\\Projects\\REAS\\GLAS\\origin_images"
des_name = "C:\\Projects\\REAS\\GLAS\\32x32"
if not os.path.exists(des_name):
    os.makedirs(des_name)
FileList = []
DesFileList = []
for dirname in os.listdir(name):  # [1:] Excludes .DS_Store from Mac OS
    path = os.path.join(name, dirname)
    directory = os.path.join(des_name, dirname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            FileList.append(os.path.join(name, dirname, filename))
            DesFileList.append(os.path.join(directory, filename))

for i in range(len(FileList)):
    print(FileList[i])
    print(DesFileList[i])
    try:
        img = Image.open(FileList[i])
        if img.size[0] != basewidth or img.size[1] != basewidth:
            #wpercent = (basewidth / float(img.size[0]))
            #hsize = int((float(img.size[1]) * float(wpercent)))
            #hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)
            img.save(DesFileList[i])
    except IOError:
        print("Failed")
