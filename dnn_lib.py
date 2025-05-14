import math

import h5py
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.io
import sklearn

use_random_seed = True

def load_dataset():
    train_dataset = h5py.File('./cat-images-classification-dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('./cat-images-classification-dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_dataset_sklearn():
    """
    sklearn.datasets.make_circles(n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8)

    Make a large circle containing a smaller circle in 2d.
    A simple toy dataset to visualize clustering and classification algorithms.

    returns:
    x - ndarray of shape (n_samples, 2) -The generated samples.
    y - ndarray of shape (n_samples,) - The integer labels (0 or 1) for class membership of each sample.
    """
    if use_random_seed: np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    if use_random_seed: np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.figure()
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    plt.show()
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def load_2D_dataset():
    # Load data
    data = scipy.io.loadmat('datasets/data.mat')

    # Split data
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    # Visualize data
    plt.scatter(train_X[0,:], train_X[1, :], c=train_Y, s = 40, cmap=plt.cm.Spectral)

    return train_X, train_Y, test_X, test_Y


def load_dataset_moons():
    plt.figure()
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    plt.draw()

    return train_X, train_Y


def initialize_parameters_shallow(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    if use_random_seed : np.random.seed(1)

    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1)) # broadcasting used later in the computation to add it to each of n_x columns
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1)) # broadcasting used later in the computation to add it to each of n_h columns

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters

def initialize_parameters_deep_xavier(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    if use_random_seed == True: np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # number of layers in the network (including 0th (input) layer)

    for l in range(1, L): # range returns 1,2,3,...,L-2,L-1 (it's ok as we're counting layers from 1)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def initialize_parameters_deep_he(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    if use_random_seed == True: np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network (including 0th (input) layer)

    for l in range(1, L): # range returns 1,2,3,...,L-2,L-1 (it's ok as we're counting layers from 1)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_step_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    linear_cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W,A_prev) + b

    assert (Z.shape == (W.shape[0], A_prev.shape[1]))

    return Z

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    """
    A = 1 / (1 + np.exp(-Z) )
    assert (A.shape == Z.shape)

    return A

def relu(Z):
    """
    Implement the RELU activation function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """
    A = np.maximum(0,Z)
    assert (A.shape == Z.shape)

    return A

def activation_step_forward(Z, activation):
    """
    Implements the forward activation step in neuron

    Arguments:
    Z -- numpy array of any shape
    activation -- string defining the activation function ("sigmoid", "relu", "tanh - Not implemented yet")

    Returns:
    A -- output of the activation function, same shape as Z
    activation_cache -- returns Z as well, useful during backpropagation
    """
    A = np.zeros((Z.shape[0],Z.shape[1]))

    if activation == "sigmoid":
        A = sigmoid(Z)

    elif activation == "relu":
        A = relu(Z)
    else:
        print("\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter")

    return A

def single_layer_forward(A_prev, W, b, activation, keep_prob=1.):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    layer_cache -- a python tuple containing "linear_cache" and "activation_cache" for the given layer;
             stored for computing the backward pass efficiently
    """

    single_layer_cache = ()

    Z = linear_step_forward(A_prev, W, b)
    A = activation_step_forward(Z, activation)

    assert (A.shape == (W.shape[0], A_prev.shape[1])) # Z.shape

    if keep_prob == 1.:
        single_layer_cache = (Z, A, W, b, A_prev)
    elif keep_prob < 1. and keep_prob > 0.:
        D = (np.random.rand(A.shape[0],A.shape[1]) < keep_prob).astype(int) # dropout draw
        A = np.multiply(A,D) / keep_prob # dropout and inverted dropout
        single_layer_cache = (Z, A, W, b, A_prev, D)
    else:
        print("\033[91mError! Please select valid keep_prob value")

    return A, single_layer_cache

def L_layer_model_forward(X, parameters, keep_prob=None):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """
    L = len(parameters) // 2
    A_prev = X
    model_cache = {}

    for l in range(1,L): # layer 1 to L-1
        if keep_prob == None:
            A, single_layer_cache = single_layer_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        else:
            A, single_layer_cache = single_layer_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                                         "relu", keep_prob=keep_prob[l - 1])
        model_cache["layer" + str(l)] = single_layer_cache
        A_prev = A

    AL, single_layer_cache = single_layer_forward(A_prev, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    model_cache["layer" + str(L)] = single_layer_cache

    return AL, model_cache


def compute_cross_entropy_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost_cross_entropy -- cross-entropy cost
    """
    m = Y.shape[1]

    # AL Pre-processing to get rid of inf and nan when using drop-out and AL could be 0 if b is not learned
    AL[AL == 0.] = 1e-10
    AL[AL == 1.] = 1. - 1e-10

    cost_cross_entropy = - (np.dot(Y,np.log(AL).T) + np.dot(1-Y,np.log(1-AL).T))
    cost_cross_entropy = np.squeeze(cost_cross_entropy) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost_cross_entropy

def compute_L2_regularization_cost(m, parameters, lambd):
    """
    Implement the cost function of L2 regularization.

    Arguments:
    m -- number of examples
    parameters -- python dictionary containing parameters of the model
    lambd -- regularization hyperparameter

    Returns:
    cost_L2_regularization - value of the L2 norm regularization cost
    """

    L = len(parameters) // 2
    L2_norm = 0
    for l in range(1,L+1):
        L2_norm += np.sum(np.square(parameters['W'+str(l)]))
    cost_L2_regularization = lambd/2 * L2_norm

    return cost_L2_regularization


def compute_cost(AL, Y, m, parameters, lambd):
    """

    :param AL:
    :param Y:
    :param m:
    :param parameters:
    :param lambd:
    :return:
    """

    cost = 0

    if lambd == 0.:
        cost = compute_cross_entropy_cost(AL, Y)
    elif lambd > 0.:
        cost = compute_cross_entropy_cost(AL, Y) + compute_L2_regularization_cost(m, parameters, lambd)
    else:
        print("\033[91mError! Please select valid lambd value")

    return cost


def linear_step_backward(dZ, A_prev, W, b, lambd=0., keep_prob=1.0):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    m = dZ.shape[1] # A_prev.shape[1] would be ok as well, as m is common for Z,dZ,A,dA and is common for all layers

    if lambd == 0.:
        dW = 1./m * np.dot(dZ,A_prev.T) # dW.shape = (n_l, n_(l-1)), dZ.shape = (n_l, m), A_prev.shape = (n_(l-1), m)
    elif lambd > 0.:
        dW = 1./m * np.dot(dZ,A_prev.T) + lambd/m * W # L2 norm part of the cost derivative added
    else:
        print("\033[91mError! Please select valid lambd value")

    db = 1./m * np.sum(dZ, axis=1, keepdims=True) # db.shape = (n_1, 1),  dZ.shape = (n_l, m)
    dA_prev = np.dot(W.T,dZ) # no division by number of training examples (m) as there is no sum over the m examples
    # dA_prev.shape = (n_(l-1), m), dW.shape = (n_l, n_(l-1)), dZ.shape = (n_l, m)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    activation_cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    a = sigmoid(Z)
    dZ = dA * a * (1-a) # elementwise multiplication

    assert (dZ.shape == Z.shape)

    return dZ

def relu_backward(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    activation_cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    # dZ = np.multiply(dA,Z > 0) # if Z > 0 then g'(Z) = 1 else if z <= 0 then g'(Z) = 0 | then we multiply elementwise

    # Alternate code, same behavior
    dZ = np.array(dA, copy=True)  # copying dA to dZ.
    dZ[Z <= 0] = 0 # When z <= 0, you should set dz to 0 as well.

    assert (dZ.shape == Z.shape)

    return dZ

def activation_step_backward(dA, Z, activation):
    """
    Implement the backward propagation for a single activation step.

    Arguments:
    dA -- post-activation gradient, of any shape
    activation_cache -- 'Z' where we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.zeros(dA.shape)

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)

    elif activation == "relu":
        dZ = relu_backward(dA, Z)

    else:
        print("\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter")

    assert (dZ.shape == dA.shape)

    return dZ

def single_layer_backward(dA, single_layer_cache, activation, lambd=0., keep_prob=1.):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    if keep_prob == 1.:
        Z, A, W, b, A_prev = single_layer_cache
    elif keep_prob < 1. and keep_prob > 0.:
        Z, A, W, b, A_prev, D = single_layer_cache
        dA = np.multiply(dA, D) / keep_prob
    else:
        print("\033[91mError! Please select valid keep_prob value")

    dZ = activation_step_backward(dA, Z, activation)
    dA_prev, dW, db = linear_step_backward(dZ, A_prev, W, b, lambd=lambd, keep_prob= keep_prob)

    return dA_prev, dW, db

def L_layer_model_backward(AL, Y, model_cache, lambd=0., keep_prob=None):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    model_cache -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(model_cache)

    # Last layer backpropagation
    dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    # grads["dA" + str(L)] = dAL

    single_layer_cache = model_cache["layer" + str(L)]
    dA_prev, dW, db = single_layer_backward(dAL, single_layer_cache, activation="sigmoid", lambd=lambd)
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    # grads["dA" + str(L-1)] = dA_prev

    # Hidden layers backpropagation
    for l in reversed(range(1,L)): # Loop from L-1 to 1
        single_layer_cache = model_cache["layer" + str(l)]
        if keep_prob == None:
            dA_prev, dW, db = single_layer_backward(dA_prev, single_layer_cache, activation="relu",
                                                lambd=lambd)
        else:
            dA_prev, dW, db = single_layer_backward(dA_prev, single_layer_cache, activation="relu",
                                                lambd=lambd, keep_prob=keep_prob[l-1])
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
        # grads["dA" + str(l - 1)] = dA_prev

    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # 2 parameters per 1 layer

    for l in range(1, L+1): # layers 1 to L
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

def shallow_model_train(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    if use_random_seed : np.random.seed(1)
    # n_x, n_h, n_y = layers_dims
    costs = []
    parameters = initialize_parameters_shallow(layers_dims[0], layers_dims[1], layers_dims[2])


    for i in range(0,num_iterations) :
        model_cache = {}
        grads = {}

        A1, model_cache["layer1"] = single_layer_forward(X, parameters["W1"], parameters["b1"], "relu")
        AL, model_cache["layer2"] = single_layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

        cost = compute_cross_entropy_cost(AL, Y)

        dAL = - np.divide(Y,AL) + np.divide(1-Y,1-AL)

        grads["dA1"], grads["dW2"], grads["db2"] = single_layer_backward(dAL, model_cache["layer2"], "sigmoid")
        grads["dA0"], grads["dW1"], grads["db1"] = single_layer_backward(grads["dA1"], model_cache["layer1"], "relu")


        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and ( i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)

    return parameters, costs

def train_deep_fully_connected_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, mini_batch_size = None, print_cost=False, initialization ="he", lambd=0., keep_prob = None, gradient_verification=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    initialization -- initialization method - 'he' or 'xavier'
    lambd -- L2 regularization hyperparameter, scalar
    keep_prob -- dropout hyperparameter, scalar

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    if use_random_seed : np.random.seed(1)
    L = len(layers_dims)
    m = Y.shape[1]
    grads = {}
    model_cache = {}
    costs = []
    parameters = {}

    if initialization == "he":
        parameters = initialize_parameters_deep_he(layers_dims)
    elif initialization == "xavier":
        parameters = initialize_parameters_deep_xavier(layers_dims)
    else:
        print ("\033[91mError! Please select correct initialization method")

    for i in range(num_iterations):

        mini_batches = []
        if mini_batch_size is None:
            mini_batches.append((X, Y))
        elif mini_batch_size > 0:
            mini_batches = generate_mini_batches(X, Y, mini_batch_size)
        else:
            print("\033[91mError! Please provide a proper mini batch size")

        cost_total = 0
        for mini_batch in mini_batches:

            (mini_batch_X, mini_batch_Y) = mini_batch

            AL, model_cache = L_layer_model_forward(mini_batch_X, parameters, keep_prob=keep_prob)

            cost_total += compute_cost(AL, mini_batch_Y, m, parameters, lambd)

            grads = L_layer_model_backward(AL, mini_batch_Y, model_cache, lambd=lambd, keep_prob=keep_prob)

            if gradient_verification and i % 5000 == 0:
                verify_gradient(grads, mini_batch_X, mini_batch_Y, layers_dims, parameters, model_cache, lambd=lambd, keep_prob=keep_prob)

            parameters = update_parameters(parameters, grads, learning_rate)

        cost_average = cost_total / m

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost_average)))
        if i % 100 == 0:
            costs.append(cost_average)




    return parameters, costs

def plot_costs(costs, learning_rate=0.0075):
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def predict(X, parameters, prediction_threshold=0.5):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    prediction_threshold -- threshold above which the probability is enough to claim the true prediction

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m)) # placeholder for predictions

    # Forward propagation
    probabilities, caches = L_layer_model_forward(X, parameters) # forward model returns the probabilities that X is True (cat in this case)

    # convert probabs to 0/1 predictions
    p = probabilities > prediction_threshold
    # for i in range(0, probabilities.shape[1]):
    #     if probabilities[0, i] > prediction_threshold:
    #         p[0, i] = 1
    #     else:
    #         p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))

    return p

def calculate_accuracy(p, y):
    """
    This function is used to calculate the accuracy of the prediction compared to ground truth.

    Arguments:
    p -- predictions for the given dataset
    y -- true labels for the given dataset

    Returns:
    accuracy -- accuracy of the given predictions
    """
    m = y.shape[1]
    accuracy = np.sum((p == y) / m)

    return accuracy

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    sum = p + y # 0+0 = 0, 1+1 = 2, 0+1 and 1+0 = 1 - if mismatch
    mislabeled_indices = np.asarray(np.where(sum == 1))
    num_images = len(mislabeled_indices[0])


    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    plt.figure()
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode("utf-8"))

def plot_decision_boundary(parameters, X, Y, padding = 1):
    """
    Script to plot the decisions boundary.
    Creates a very dense grid of points through the entire possible range and predict the prediction based on location to create a map.
    Add a reference set based on X and Y to visualise the predictions and accuracy

    :param parameters: Trained parameters of the model
    :param X: Position of the points
    :param Y: Classification of the points
    """
    plt.figure()
    plt.title('Decision map')
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])

    # Set min and max values and add some padding
    x_min = X[0, :].min() - padding
    x_max = X[0, :].max() + padding
    y_min = X[1, :].min() - padding
    y_max = X[1, :].max() + padding

    h = 0.01
    ### Generate the spacing on x any y axes
    # np.arrange - Return evenly spaced values within a given interval.
    # arange(start, stop, step) - Values are generated within the half - open interval[start, stop), with spacing between values given by step.
    x_spacing = np.arange(x_min, x_max, h)
    y_spacing = np.arange(y_min, y_max, h)

    ### Generate a grid of points separated by distance h
    # xx, yy = np.meshgrid(x, y)
    # x - 1D vector of x coordinates (nx, )
    # y - 1D vector of y coordinates (ny, )
    # xx - 2d vector containing the x coordinates for specific points (ny, nx)
    # yy - 2d vector containing the y coordinates for specific points (ny, nx)
    xx, yy = np.meshgrid(x_spacing, y_spacing)

    ### Preprocess the grid point to match the model input
    # np.c_ - column stack - allows to concatenate 1-D arrays into 2-D matrix (column next to column)
    # Transpose needed to convert (m,2) -> (2,m)
    # np.r_ is more complicated to use
    # np.ravel converts matrix into vector element by element (C-style order make last axis change fastest [0][k] and 1st axis change slowest [1][k])
    grid_X = np.c_[xx.ravel(order='C'), yy.ravel(order='C')].T

    ### Perform Prediction for the entire meshgrid
    grid_predictions = predict(grid_X, parameters)

    ### Backprocess the predictions to match the grid format
    grid_predictions = np.reshape(grid_predictions, xx.shape, order='C')

    ### Plot all the grid points to create a decision map
    # plt.contourf function is faster then scatter and has slightly different colors (that why backprocessing)
    plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Spectral)
    # plt.scatter(xx.ravel(), yy.ravel(), c=Z.ravel(), cmap=plt.cm.Spectral) # works, but slower

    ### Plot the dataset with predictions on top of decision map to compare the accuracy
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)

    ### axes limits
    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])

    ### axis labels
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def dictionary_to_vector(parameters):
    """

    :param parameters:
    :return:
    """

    theta = []

    count = 0
    for key in parameters:
        new_vector = np.reshape(parameters[key],(-1,1))

        if count == 0:
            theta = new_vector
            count += 1
        else:
            theta = np.concatenate((theta, new_vector), axis=0)

    return theta


def gradients_to_vector(gradients, layer_dims):
    """

    :param parameters:
    :return:
    """

    gradients_vector = []
    L = len(layer_dims)  # number of layers in the network (including 0th (input) layer)


    init = False
    for l in range(1, L):  # range returns 1,2,3,...,L-2,L-1 (it's ok as we're counting layers from 1)

        new_vector_dW = np.reshape(gradients['dW' + str(l)], (-1, 1))
        assert (new_vector_dW.shape == (layer_dims[l] * layer_dims[l - 1], 1))

        new_vector_db = np.reshape(gradients['db' + str(l)], (-1, 1))
        assert (new_vector_db.shape == (layer_dims[l], 1))

        if (not init):
            gradients_vector = new_vector_dW
            gradients_vector = np.concatenate((gradients_vector, new_vector_db), axis=0)
            init = True
        else:
            gradients_vector = np.concatenate((gradients_vector, new_vector_dW), axis=0)
            gradients_vector = np.concatenate((gradients_vector, new_vector_db), axis=0)

    return gradients_vector


def vector_to_dictionary(updated_vector, original_dictionary):
    """

    :param updated_vector:
    :param original_dictionary:
    :return:
    """

    updated_dictionary = {}

    starting_index = 0
    for key in original_dictionary:
        updated_dictionary[key] = updated_vector[starting_index: starting_index + original_dictionary[key].shape[0] * original_dictionary[key].shape[1]].reshape((original_dictionary[key].shape[0], original_dictionary[key].shape[1]))
        starting_index += original_dictionary[key].shape[0] * original_dictionary[key].shape[1]

    return updated_dictionary


def drop_parameters(theta_plus, model_cache, layer_dims):

    L = len(layer_dims)

    for l in range(1,L):
        single_layer_cache = model_cache["layer" + str(l)]
        Z, A, W, b, A_prev, D = single_layer_cache

        




def calculate_gradients_derivative_approximations(X, Y, parameters, model_cache, layer_dims, lambd=0., keep_prob=None, epsilon=1e-7):
    """

    :param X:
    :param Y:
    :param parameters:
    :param lambd:
    :param keep_prob:
    :param epsilon:
    :return:
    """

    m = Y.shape[1]

    theta = dictionary_to_vector(parameters)
    num_parameters = theta.shape[0]

    cost_plus = np.zeros((num_parameters,1))
    cost_minus = np.zeros((num_parameters,1))
    gradients_derivative_approximations = np.zeros((num_parameters,1))

    for i in range(num_parameters):

        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)

        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        if keep_prob:
            # theta_plus = drop_parameters(theta_plus, model_cache, layer_dims)
            # theta_minus = drop_parameters(theta_minus, model_cache, layer_dims)
            # maybe it needs to be developed inside L_layer_model_forward, where D is taken from model_cache
            print("\033[93m" + "Dropout gradient verification to be developed " + "\033[0m")

        AL_plus, _ = L_layer_model_forward(X, vector_to_dictionary(theta_plus, parameters), keep_prob=keep_prob)
        cost_plus[i] = compute_cost(AL_plus, Y, m, vector_to_dictionary(theta_plus, parameters), lambd=lambd)

        AL_minus, _ = L_layer_model_forward(X, vector_to_dictionary(theta_minus, parameters), keep_prob=keep_prob)
        cost_minus[i] = compute_cost(AL_minus, Y, m, vector_to_dictionary(theta_minus, parameters), lambd=lambd)

        gradients_derivative_approximations[i] = (cost_plus[i] - cost_minus[i]) / (2. * epsilon)

    return gradients_derivative_approximations


def calculate_relative_difference(gradients_backward_values, gradients_derivative_approximations):
    """
    Calculate relative difference between values from backward propagation and derivative approximation

    Arguments:
    :param gradients_backward_values: values of gradients calculated by formulas in backward propagation
    :param gradients_derivative_approximations: values fo gradients calculated using derivative approximation

    Returns:
    :return: relative_difference: relative difference between values from backward propagation and derivative approximation
    """

    nominator = np.linalg.norm(gradients_backward_values - gradients_derivative_approximations)
    denominator = np.linalg.norm(gradients_backward_values) + np.linalg.norm(gradients_derivative_approximations)

    relative_difference = nominator / denominator

    return relative_difference


def verify_gradient(gradients_backward_values, X, Y, layer_dims, parameters, model_cache, lambd=0., keep_prob=None, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters
    X -- input datapoint, of shape (input size, number of examples)
    Y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    gradients_derivative_approximations = calculate_gradients_derivative_approximations(X, Y, parameters, model_cache, layer_dims, lambd, keep_prob, epsilon)
    relative_difference = calculate_relative_difference(gradients_to_vector(gradients_backward_values, layer_dims), gradients_derivative_approximations)

    if relative_difference > 2e-7:
        print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
            relative_difference) + "\033[0m")
    else:
        print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
            relative_difference) + "\033[0m")


def generate_mini_batches(X,Y,mini_batch_size):
    """
    Creates a list of random mini batches from(X,Y)

    Arguments:
    X -- input data, of shape(input size,number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1,number of examples)
    mini_batch_size -- size of the mini-batches,integer

    Returns:
    mini_batches--list of synchronous (mini_batch_X, mini_batch_Y)
    """

    mini_batches = []
    m = Y.shape[1]

    #Shuffle the training data
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    #Split the shuffled data into mini_batches
    num_complete_mini_batches = math.floor(m / mini_batch_size)
    for k in range(num_complete_mini_batches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k+1) * mini_batch_size]

        mini_batch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    #Checkforincompleteminibatchattheend
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_mini_batches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_mini_batches * mini_batch_size: m]

        mini_batch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
