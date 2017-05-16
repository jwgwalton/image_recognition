import numpy as np


class NeuralNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #randomly initialise biases
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #randomly initialise weights

    def evaluate_network(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def optimise_network(self, training_data, test_data, runs=100, learning_rate=1.0):
        n_test = len(test_data)
        for j in xrange(runs):
            self.train_network(training_data, learning_rate)
            print "Run {0}: {1} / {2}".format(
                j, self.evaluate_validation_images(test_data), n_test)

    def train_network(self, training_data, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in training_data:
            delta_error_b, delta_error_w = self.back_propogate(x, y) #calculate the gradient for all values
            nabla_b = [nb+deb for nb, deb in zip(nabla_b, delta_error_b)]
            nabla_w = [nw+dew for nw, dew in zip(nabla_w, delta_error_w)]

        # gradient descent to update weights and biases
        self.weights = [weight - (learning_rate / len(training_data)) * nw for weight, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(training_data)) * nb for b, nb in zip(self.biases, nabla_b)]

    def back_propogate(self, x, y):
        error_b = [np.zeros(b.shape) for b in self.biases]
        error_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        z_vectors = [] # list of zs for each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_deriv(z_vectors[-1])
        error_b[-1] = delta
        error_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = z_vectors[-l]
            sp = sigmoid_deriv(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            error_b[-l] = delta
            error_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return error_b, error_w

    def evaluate_validation_images(self, test_data):
        test_results = [(np.argmax(self.evaluate_network(x)), y)for (x, y) in test_data] # returns highest value from array and as such highest match
        return sum(int(x == y) for (x, y) in test_results) #number of results evaluated correctly

    def cost_derivative(self, output_activations, y):
        # return partial derivative of cost function for outputs
        return output_activations-y


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))
