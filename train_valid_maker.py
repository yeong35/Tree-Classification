from itertools import count
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# file path
# 낙엽송 
larch = "./larch/"
# 잣나무
pine = "./pine/"

# slow is 0, fast is 1
def make_dataframe(path, species):
    
    fname_list = [file for file in os.listdir(path) if file.endswith(".ply")]

    X_train, X_val= train_test_split(fname_list, test_size=0.2, random_state=42)
    X_val, X_test= train_test_split(X_val, test_size=0.5, random_state=42)

    X_train = [species+"/"+i+"\n" for i in X_train]
    X_val = [species+"/"+i+"\n" for i in X_val]
    X_test = [species+"/"+i+"\n" for i in X_test]
    return X_train, X_val, X_test
    

larch_train, larch_val, larch_test = make_dataframe(larch, "larch")
pine_train, pine_val, pine_test = make_dataframe(pine, "pine")

# train file
f = open("./Sample/train.txt", 'w')
for i in larch_train:
    f.write(i)

for i in pine_train:
    f.write(i)
f.close()

# valid file
f = open("./Sample/valid.txt", 'w')
for i in larch_val:
    f.write(i)

for i in pine_val:
    f.write(i)
f.close()

# test file
f = open("./Sample/test.txt", 'w')
for i in larch_test:
    f.write(i)

for i in pine_test:
    f.write(i)
f.close()

print("Done")