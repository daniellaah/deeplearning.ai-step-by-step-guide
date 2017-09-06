import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
%matplotlib inline

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

X_train = train_set_x_flatten/255
y_train = train_set_y
X_test = test_set_x_flatten/255
y_test = test_set_y
# Please note that the above code is from the programming assignment
import LogisticRegression
num_iter = 2001
learning_rate = 0.005
clf = LogisticRegression().fit(X_train, y_train, num_iter, learning_rate, True, 500)
train_acc = clf.accuracy_score(X_train, y_train)
print('training acc: {}'.format(train_acc))
test_acc = clf.accuracy_score(X_test, y_test)
print('testing acc: {}'.format(test_acc))

# output:
# Cost after iteration 0: 0.693147
# Cost after iteration 500: 0.303273
# Cost after iteration 1000: 0.214820
# Cost after iteration 1500: 0.166521
# Cost after iteration 2000: 0.135608
# training acc: 0.9904306220095693
# testing acc: 0.7
