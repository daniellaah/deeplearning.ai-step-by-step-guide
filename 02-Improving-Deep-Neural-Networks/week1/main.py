from utils import load_dataset
from DeepNeuralNetwork import DeepNeuralNetwork
import numpy as np

if __name__ == '__main__':
    # ----------------------------load data-------------------------
    train_X, train_Y, test_X, test_Y = load_dataset(plot_data=False)

    layers_dims = [train_X.shape[0], 10, 5, 1]
    activations = ['relu', 'relu', 'sigmoid']
    np.seterr(all='ignore')
    # --------------------------zero init----------------------------
    dnn = DeepNeuralNetwork(layers_dims, activations, init='zero')
    dnn = dnn.fit(train_X, train_Y, 15000, 0.01, print_cost=False, print_num=1000)
    predictions_train = dnn.accuracy_score(train_X, train_Y)
    print ("zero init accuracy on train: {:.2f}".format(predictions_train))
    predictions_test = dnn.accuracy_score(test_X, test_Y)
    print ("zero init accuracy on test: {:.2f}".format(predictions_test))

    # ------------------------large rand init-------------------------
    dnn = DeepNeuralNetwork(layers_dims, activations, init='large')
    dnn = dnn.fit(train_X, train_Y, 15000, 0.01, print_cost=False, print_num=1000)
    predictions_train = dnn.accuracy_score(train_X, train_Y)
    print ("large random init accuracy on train: {:.2f}".format(predictions_train))
    predictions_test = dnn.accuracy_score(test_X, test_Y)
    print ("large random init accuracy on test: {:.2f}".format(predictions_test))

    # -----------------------------he init-----------------------------
    dnn = DeepNeuralNetwork(layers_dims, activations, init='he')
    dnn = dnn.fit(train_X, train_Y, 15000, 0.01, print_cost=False, print_num=1000)
    predictions_train = dnn.accuracy_score(train_X, train_Y)
    print ("he init accuracy on train: {:.2f}".format(predictions_train))
    predictions_test = dnn.accuracy_score(test_X, test_Y)
    print ("he init accuracy on test: {:.2f}".format(predictions_test))
