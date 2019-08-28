# split datasets into ten fold for conducting cross validation
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


input_fr = open(os.path.join('..', 'seq2seq', 'data', 'single_review_reply.txt'))
X = np.array(input_fr.readlines())
input_fr.close()
y = np.zeros(len(X))
kf = KFold(n_splits=10,shuffle=True)

count = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(len(X_test), len(y_test))
    data_fw = open(os.path.join('.', 'data', 'data_part_' + str(count) + '.txt'), 'w')
    data_fw.writelines(X_test)
    data_fw.close()
    count += 1




# for i in range(10):
#     data_fw = open(os.path.join('.', 'data', 'data_part_'+ str(i) +'.txt'), 'w')
#     X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)
#     data_fw.writelines(X_test)
#     data_fw.close()
