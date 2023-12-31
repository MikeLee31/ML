import numpy as np
import torch
import re

def test_device():
    
    #cuda是否可用,True or False
    ava = torch.cuda.is_available()
    print(ava)
    # 返回gpu数量；
    count = torch.cuda.device_count()
    print(count)
    # 返回gpu名字，设备索引默认从0开始；
    name = torch.cuda.get_device_name(0)
    print(name)
    # 返回当前设备索引；
    index = torch.cuda.current_device()
    print(index)



if __name__ == '__main__':
    test_device()
