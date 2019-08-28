# split dataset into training and test dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split

source_data = open("/research/lyu1/cygao/workspace/data/review_reply_new_new.txt")
train_data_fw = open("/research/lyu1/cygao/workspace/data/train_data.txt", "w")
valid_data_fw = open("/research/lyu1/cygao/workspace/data/valid_data.txt", "w")
test_data_fw = open("/research/lyu1/cygao/workspace/data/test_data.txt", "w")
lines = source_data.readlines()
source_data.close()
y = np.zeros(len(lines))

X_train, X_test, y_train, y_test = train_test_split(lines, y, test_size=0.05, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, shuffle=True)

train_data_fw.writelines(X_train)
train_data_fw.close()
valid_data_fw.writelines(X_val)
valid_data_fw.close()
test_data_fw.writelines(X_test)
test_data_fw.close()