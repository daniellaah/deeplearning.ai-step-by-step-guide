def sigmoid(z):
    return 1. / (1.+np.exp(-z))

class LogisticRegression():
    def __init__(self):
        pass

    def __parameters_initializer(self, input_size):
        # initial parameters with zeros
        w = np.zeros((input_size, 1), dtype=float)
        b = 0.0
        return w, b

    def __forward_propagation(self, X):
        m = X.shape[1]
        A = sigmoid(np.dot(self.w.T, X) + self.b)
        return A

    def __compute_cost(self, A, Y):
        m = A.shape[1]
        cost = -np.sum(Y*np.log(A) + (1-Y)*(np.log(1-A))) / m
        return cost

    def cost_function(self, X, Y):
        # use the result from forward propagation and the label Y to compute cost
        A = self.__forward_propagation(X)
        cost = self.__compute_cost(A, Y)
        return cost

    def __backward_propagation(self, A, X, Y):
        m = X.shape[1]
        # backward propagation computes gradients
        dw = np.dot(X, (A-Y).T) / m
        db = np.sum(A-Y) / m
        grads = {"dw": dw, "db": db}
        return grads

    def __update_parameters(self, grads, learning_rate):
        self.w -= learning_rate * grads['dw']
        self.b -= learning_rate * grads['db']

    def fit(self, X, Y, num_iterations, learning_rate, print_cost=False, print_num=100):
        self.w, self.b = self.__parameters_initializer(X.shape[0])
        for i in range(num_iterations):
            # forward_propagation
            A = self.__forward_propagation(X)
            # compute cost
            cost = self.__compute_cost(A, Y)
            # backward_propagation
            grads = self.__backward_propagation(A, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            # update parameters
            self.__update_parameters(grads, learning_rate)
            # print cost
            if i % print_num == 0 and print_cost:
                print ("Cost after iteration {}: {:.6f}".format(i, cost))
        return self

    def predict_prob(self, X):
        # result of forward_propagation is the probability
        A = self.__forward_propagation(X)
        return A

    def predict(self, X, threshold=0.5):
        pred_prob = self.predict_prob(X)
        threshold_func = np.vectorize(lambda x: 1 if x > threshold else 0)
        Y_prediction = threshold_func(pred_prob)
        return Y_prediction

    def accuracy_score(self, X, Y):
        pred = self.predict(X)
        return len(Y[pred == Y]) / Y.shape[1]
