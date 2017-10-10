import PIL
from PIL import Image
import os
from array import *
from random import shuffle
import re

basewidth = 32

#Names = [['./training-images','train'], ['./test-images','test']]
Names = [['./small_set','train']]
for name in Names:

    FileList = []
    for dirname in os.listdir(name[0]):  # [1:] Excludes .DS_Store from Mac OS
        path = os.path.join(name[0], dirname)
        for filename in os.listdir(path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                FileList.append(os.path.join(name[0], dirname, filename))

    shuffle(FileList)  # Usefull for further segmenting the validation set

    for filename in FileList:
        img = Image.open(filename)
        if img.size[0] != basewidth or img.size[1] != basewidth:
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
            img.save(filename)
            print(filename)
