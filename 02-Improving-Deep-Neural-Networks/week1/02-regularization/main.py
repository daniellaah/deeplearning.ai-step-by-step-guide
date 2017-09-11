import numpy as np
import matplotlib.pyplot as plt
from utils import load_2D_dataset, plot_decision_boundary
from DeepNeuralNetwork import DeepNeuralNetwork

if __name__ == '__main__':
    np.seterr(all='ignore')
    # load data
    train_X, train_Y, test_X, test_Y = load_2D_dataset(False)

    layers_dims = [train_X.shape[0], 20, 3, 1]
    activations = ['relu', 'relu', 'sigmoid']

    print('-----------------------without regularization------------------------')
    dnn = DeepNeuralNetwork(layers_dims, activations, init='other')
    dnn = dnn.fit(train_X, train_Y, num_iterations=30000,
                                    learning_rate=0.3,
                                    print_cost=True, print_num=10000)
    predictions_train = dnn.accuracy_score(train_X, train_Y)
    print ("accuracy on train without regularization: {:.2f}%".format(predictions_train*100))
    predictions_test = dnn.accuracy_score(test_X, test_Y)
    print ("accuracy on test without regularization: {:.2f}%".format(predictions_test*100))

    print('--------------------------l2 regularization--------------------------')
    dnn_l2 = DeepNeuralNetwork(layers_dims, activations, init='other')
    dnn_l2 = dnn_l2.fit(train_X, train_Y, num_iterations=30000,
                                          learning_rate=0.3,
                                          l2=0.7, print_cost=True, print_num=10000)
    predictions_train = dnn_l2.accuracy_score(train_X, train_Y)
    print ("accuracy on train without regularization: {:.2f}%".format(predictions_train*100))
    predictions_test = dnn_l2.accuracy_score(test_X, test_Y)
    print ("accuracy on test without regularization: {:.2f}%".format(predictions_test*100))

    print('-----------------------dropout regularization-------------------------')
    keep_prob = [1, 0.86, 0.86, 1]
    dnn_dropout = DeepNeuralNetwork(layers_dims, activations, init='other')
    dnn_dropout = dnn_dropout.fit(train_X, train_Y, num_iterations=30000,
                                                    learning_rate=0.3,
                                                    l2=0,
                                                    keep_prob=keep_prob,
                                                    print_cost=True, print_num=10000)
    predictions_train = dnn_dropout.accuracy_score(train_X, train_Y)
    print ("accuracy on train without regularization: {:.2f}%".format(predictions_train*100))
    predictions_test = dnn_dropout.accuracy_score(test_X, test_Y)
    print ("accuracy on test without regularization: {:.2f}%".format(predictions_test*100))

    # plot decision boundary without regularization
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: dnn.predict(x.T), train_X, train_Y)

    # plot decision boundary with l2 regularization
    plt.title("Model with l2 regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: dnn_l2.predict(x.T), train_X, train_Y)

    # plot decision boundary with dropout regularization
    plt.title("Model with dropout regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: dnn_dropout.predict(x.T), train_X, train_Y)
