import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, plot_decision_boundary
from DeepNeuralNetwork import DeepNeuralNetwork


if __name__ == '__main__':
    np.seterr(all='ignore')
    # ----------------------------load data-------------------------
    train_X, train_Y, test_X, test_Y = load_dataset(plot_data=False)

    layers_dims = [train_X.shape[0], 10, 5, 1]
    activations = ['relu', 'relu', 'sigmoid']

    # --------------------------zero init----------------------------
    dnn1 = DeepNeuralNetwork(layers_dims, activations, init='zero')
    dnn1 = dnn1.fit(train_X, train_Y, 15000, 0.01, print_cost=False, print_num=1000)
    predictions_train = dnn1.accuracy_score(train_X, train_Y)
    print ("zero init accuracy on train: {:.2f}".format(predictions_train))
    predictions_test = dnn1.accuracy_score(test_X, test_Y)
    print ("zero init accuracy on test: {:.2f}".format(predictions_test))

    # ------------------------large rand init-------------------------
    dnn2 = DeepNeuralNetwork(layers_dims, activations, init='large')
    dnn2 = dnn2.fit(train_X, train_Y, 15000, 0.01, print_cost=False, print_num=1000)
    predictions_train = dnn2.accuracy_score(train_X, train_Y)
    print ("large random init accuracy on train: {:.2f}".format(predictions_train))
    predictions_test = dnn2.accuracy_score(test_X, test_Y)
    print ("large random init accuracy on test: {:.2f}".format(predictions_test))

    # -----------------------------he init-----------------------------
    dnn3 = DeepNeuralNetwork(layers_dims, activations, init='he')
    dnn3 = dnn3.fit(train_X, train_Y, 15000, 0.01, print_cost=False, print_num=1000)
    predictions_train = dnn3.accuracy_score(train_X, train_Y)
    print ("he init accuracy on train: {:.2f}".format(predictions_train))
    predictions_test = dnn3.accuracy_score(test_X, test_Y)
    print ("he init accuracy on test: {:.2f}".format(predictions_test))


    # plot decision boundary for zero initialization
    plt.title("Model with zero initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: dnn1.predict(x.T), train_X, train_Y)

    # plot decision boundary for large random initialization
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: dnn2.predict(x.T), train_X, train_Y)

    # plot decision boundary for he initialization
    plt.title("Model with he initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5,1.5])
    axes.set_ylim([-1.5,1.5])
    plot_decision_boundary(lambda x: dnn3.predict(x.T), train_X, train_Y)
