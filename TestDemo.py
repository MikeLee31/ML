import numpy as np
import re

images_dir = 'D:\Lee\Desktop\mindspore\images.txt'

if __name__ == '__main__':

    lineList = []
    # file = open(images_dir,'r',encoding='UTF-8')
    file = open(images_dir, 'r', encoding='UTF-8')
    line = file.readline()
    while line:
        if line.count('[') >= 1:
            line = line.replace('[', '')
            print(line)
        if line.count(']') >= 1:
            line = line.replace(']', '')
            print(line)
        lineList.append(line)
        line = file.readline()

    file = open(r'D:\target.txt', 'w', encoding='UTF-8')
    for i in lineList:
        file.write(i)
    print('ok')
