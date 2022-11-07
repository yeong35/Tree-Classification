from itertools import count
import os
import numpy as np
import pandas as pd

# file path
# 낙엽송
larch = "../Dataset/Sample/larch/"
# 잣나무
pine = "../Dataset/Sample/pine/"

# slow is 0, fast is 1
def make_dataframe(path, train_valid, species):
    fname_list = os.listdir(path)
   
    fname_list = [species+"/"+i+"\n" for i in fname_list]
    return fname_list
    

larch_txt = make_dataframe(larch, "train", "larch")
pine_txt = make_dataframe(pine, "train", "pine")

print(larch_txt)

f = open("../Dataset/Sample/trainval.txt", 'w')
for i in larch_txt:
    f.write(i)

for i in pine_txt:
    f.write(i)
f.close()

print("Done")