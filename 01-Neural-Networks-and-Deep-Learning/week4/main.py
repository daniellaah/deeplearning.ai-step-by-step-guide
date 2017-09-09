import time
import h5py
import numpy as np

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
# Please note that the above code is from the programming assignment

from DeepNeuralNetwork import DeepNeuralNetwork
layers_dims = (12288, 20, 7, 5, 1)
# layers_dims = (12288, 10, 1)
# layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
activations = ['relu', 'relu', 'relu','sigmoid']
num_iter = 2500
learning_rate = 0.0075

clf = DeepNeuralNetwork(layers_dims, activations)\
            .fit(train_x, train_y, num_iter, learning_rate, True, 100)
print('train accuracy: {:.2f}%'.format(clf.accuracy_score(train_x, train_y)*100))
print('test accuracy: {:.2f}%'.format(clf.accuracy_score(test_x, test_y)*100))

# output
# Number of training examples: 209
# Number of testing examples: 50
# Each image is of size: (64, 64, 3)
# train_x_orig shape: (209, 64, 64, 3)
# train_y shape: (1, 209)
# test_x_orig shape: (50, 64, 64, 3)
# test_y shape: (1, 50)
# train_x's shape: (12288, 209)
# test_x's shape: (12288, 50)
# Cost after iteration 0: 0.771749
# Cost after iteration 100: 0.672053
# Cost after iteration 200: 0.648263
# Cost after iteration 300: 0.611507
# Cost after iteration 400: 0.567047
# Cost after iteration 500: 0.540138
# Cost after iteration 600: 0.527930
# Cost after iteration 700: 0.465477
# Cost after iteration 800: 0.369126
# Cost after iteration 900: 0.391747
# Cost after iteration 1000: 0.315187
# Cost after iteration 1100: 0.272700
# Cost after iteration 1200: 0.237419
# Cost after iteration 1300: 0.199601
# Cost after iteration 1400: 0.189263
# Cost after iteration 1500: 0.161189
# Cost after iteration 1600: 0.148214
# Cost after iteration 1700: 0.137775
# Cost after iteration 1800: 0.129740
# Cost after iteration 1900: 0.121225
# Cost after iteration 2000: 0.113821
# Cost after iteration 2100: 0.107839
# Cost after iteration 2200: 0.102855
# Cost after iteration 2300: 0.100897
# Cost after iteration 2400: 0.092878
# train accuracy: 98.56%
# test accuracy: 80.00%
