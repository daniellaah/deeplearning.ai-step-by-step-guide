# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
%matplotlib inline

np.random.seed(1) # set a seed so that the results are consistent
X, Y = load_planar_dataset()
# Please note that the above code is from the programming assignment

import SimpleNeuralNetwork
np.random.seed(3)
num_iter = 10001
learning_rate = 1.2
input_size = X.shape[0]
hidden_layer_size = 4
clf = SimpleNeuralNetwork(input_size=input_size,
                          hidden_layer_size=hidden_layer_size)\
        .fit(X, Y, num_iter, learning_rate, True, 1000)
train_acc = clf.accuracy_score(X, Y)
print('training accuracy: {}%'.format(train_acc*100))

# output
# Cost after iteration 0: 0.693162
# Cost after iteration 1000: 0.258625
# Cost after iteration 2000: 0.239334
# Cost after iteration 3000: 0.230802
# Cost after iteration 4000: 0.225528
# Cost after iteration 5000: 0.221845
# Cost after iteration 6000: 0.219094
# Cost after iteration 7000: 0.220628
# Cost after iteration 8000: 0.219400
# Cost after iteration 9000: 0.218482
# Cost after iteration 10000: 0.217738
# training accuracy: 90.5%

for hidden_layer_size in [1, 2, 3, 4, 5, 20, 50]:
    clf = SimpleNeuralNetwork(input_size=input_size,
                               hidden_layer_size=hidden_layer_size)\
            .fit(X, Y, num_iter, learning_rate, False)
    print('{} hidden units, cost: {}, accuracy: {}%'
           .format(hidden_layer_size,
                   clf.cost_function(X, Y),
                   clf.accuracy_score(X, Y)))
# output
# 1 hidden units, cost: 0.6315593779798304, accuracy: 67.5%
# 2 hidden units, cost: 0.5727606525435293, accuracy: 67.25%
# 3 hidden units, cost: 0.2521014374551156, accuracy: 91.0%
# 4 hidden units, cost: 0.24703039056643344, accuracy: 91.25%
# 5 hidden units, cost: 0.17206481441467936, accuracy: 91.5%
# 20 hidden units, cost: 0.16003869681611513, accuracy: 92.25%
# 50 hidden units, cost: 0.16000569403994763, accuracy: 92.5%
