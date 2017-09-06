def sigmoid(z):
    return 1. / (1.+np.exp(-z))

class SimpleNeuralNetwork():
    # simple neural network with one hidden layer
    def __init__(self, input_size, hidden_layer_size):
        self.paramters = self.__parameter_initailizer(input_size, hidden_layer_size)

    def __parameter_initailizer(self, n_x, n_h):
        # W cannot be initialized with zeros
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(1, n_h) * 0.01
        b2 = np.zeros((1, 1))
        return {'W1': W1,'b1': b1,'W2': W2,'b2': b2}

    def __forward_propagation(self, X):
        W1 = self.paramters['W1']
        b1 = self.paramters['b1']
        W2 = self.paramters['W2']
        b2 = self.paramters['b2']
        # forward propagation
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        cache = {'Z1': Z1,'A1': A1,'Z2': Z2,'A2': A2}
        return A2, cache

    def __compute_cost(self, A2, Y):
        m = A2.shape[1]
        cost = -np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2)) / m
        return cost

    def cost_function(self, X, Y):
        # use the result from forward propagation and the label Y to compute cost
        A2, cache = self.__forward_propagation(X)
        cost = self.__compute_cost(A2, Y)
        return cost

    def __backward_propagation(self, cache, Y):
        A1, A2 = cache['A1'], cache['A2']
        W2 = self.paramters['W2']
        m = X.shape[1]
        # backward propagation computes gradients
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        grads = {'dW1': dW1,'db1': db1,'dW2': dW2,'db2': db2}
        return grads

    def __update_parameters(self, grads, learning_rate):
        self.paramters['W1'] -= learning_rate * grads['dW1']
        self.paramters['b1'] -= learning_rate * grads['db1']
        self.paramters['W2'] -= learning_rate * grads['dW2']
        self.paramters['b2'] -= learning_rate * grads['db2']

    def fit(self, X, Y, num_iterations, learning_rate, print_cost=False, print_num=100):
        for i in range(num_iterations):
            # forward propagation
            A2, cache = self.__forward_propagation(X)
            # compute cost
            cost = self.cost_function(X, Y)
            # backward propagation
            grads = self.__backward_propagation(cache, Y)
            # update parameters
            self.__update_parameters(grads, learning_rate)
            # print cost
            if i % print_num == 0 and print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
        return self

    def predict_prob(self, X):
        # result of forward_propagation is the probability
        A2, _ = self.__forward_propagation(X)
        return A2

    def predict(self, X, threshold=0.5):
        pred_prob = self.predict_prob(X)
        threshold_func = np.vectorize(lambda x: 1 if x > threshold else 0)
        Y_prediction = threshold_func(pred_prob)
        return Y_prediction

    def accuracy_score(self, X, Y):
        pred = self.predict(X)
        return len(Y[pred == Y]) / Y.shape[1]
