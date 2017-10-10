import PIL
from PIL import Image
import os
from array import *
from random import shuffle
import re


basewidth = 32


def save(data, file_name):
    output_file = open(file_name, 'ab+')
    data.tofile(output_file)
    output_file.close()


def process(files, data_out_file, label_out_file):
    os.remove(data_out_file) if os.path.exists(data_out_file) else None
    os.remove(label_out_file) if os.path.exists(label_out_file) else None
    data_image = array('B')
    data_label = array('B')
    count = 0
    for filename in files:
        try :
            if len(data_label) % 1000 == 999:
                save(data_image, data_out_file)
                save(data_label, label_out_file)
                count += len(data_label)
                data_image = array('B')
                data_label = array('B')

            label = 0
            if "\\NG" in filename:
                label = 1

            Im = Image.open(filename)

            pixel = Im.load()

            width, height = Im.size

            if width != basewidth or height != basewidth:
                print("Image size is not suitable: " + filename)
                continue

            for x in range(0, width):
                for y in range(0, height):
                    if type(pixel[y,x] ) == type(0):
                        data_image.append(pixel[y, x])
                        data_image.append(pixel[y, x])
                        data_image.append(pixel[y, x])
                    else:
                        data_image.append(pixel[y, x][0])
                        data_image.append(pixel[y, x][1])
                        data_image.append(pixel[y, x][2])

            data_label.append(label)  # labels start (one unsigned byte each)
        except IOError:
            print("File load error: " + filename)
    if len(data_label) > 0:
        save(data_image, data_out_file)
        save(data_label, label_out_file)
        count += len(data_label)
        data_image = array('B')
        data_label = array('B')
    print("File saved: ", count)


name = "C:\\Projects\\REAS\\GLAS\\32x32"
des_name = "C:\\Projects\\REAS\\GLAS\\BinaryData32x32"
if not os.path.exists(des_name):
    os.makedirs(des_name)
FileList = []
for dirname in os.listdir(name):  # [1:] Excludes .DS_Store from Mac OS
    path = os.path.join(name, dirname)
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            FileList.append(os.path.join(name, dirname, filename))

shuffle(FileList)

train_num = int(len(FileList) * 0.7)
test_num = len(FileList) - train_num

process(FileList[:train_num], os.path.join(des_name, "train_data.bin"), os.path.join(des_name, "train_label.bin"))
process(FileList[train_num:], os.path.join(des_name, "test_data.bin"), os.path.join(des_name, "test_label.bin"))




