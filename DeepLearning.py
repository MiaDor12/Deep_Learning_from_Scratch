"""
Introduction:
-------------

    * This file contains code for a neural network model which was built from scratch (without using sklearn).
    
    * When importing, only needs to import and use the class "Neural_Network" and not all the other functions. 
        For example:
        >>> from DeepLearning import Neural_Network
    
    * The network is limited to binary classification only.
    
    * In all hidden layers of the network the activation function that is performed is "RELU", 
      and in the output layer of the network the activation function that is performed is "Sigmoid".
      
    * Optional hyperparameters:
        - hidden_layer_sizes : The network can be in user-defined dimensions.
        - optimizer : The optimizer can be one of the following: "gradient_descent", "momentum" or "adam".
        - learning_rate : The learning rate.
        - lambd : Regularization hyperparameter.
        - batch_type : The batch_type can be one of the following: "batch", "mini_batch" or "stochastic".
        - mini_batch_size : If batch_type = "mini_batch", then need to choose also the mini_batch_size.
        - beta : Momentum hyperparameter.
        - beta1 : Exponential decay hyperparameter for the past gradients' estimates. 
        - beta2 : Exponential decay hyperparameter for the past squared gradients' estimates.
        - epsilon : Hyperparameter preventing division by zero.
        - num_epochs : Number of epochs.
        
    * The user can choose to print the cost every 1000 epochs by choosing print_cost = True, 
      and to plot a graph of the costs over the different epochs by choosing plot_cost = True.
      
    * Data dimensions:
        - X : Input data, of shape (number of features, number of examples).
        - y : True "label" vector, of shape (1, number of examples).
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy.
    
    Parameters:
    -----------
        Z : Output of the linear layer, numpy array of any shape.
    
    Returns:
    --------
        A : Output of sigmoid(Z), same shape as Z.
        cache : Returns Z as well, useful during backpropagation.
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache


def relu(Z):
    """
    Implement the RELU activation in numpy.

    Parameters:
    -----------
        Z : Output of the linear layer, numpy array of any shape.

    Returns:
    --------
        A : Output of relu(Z), of the same shape as Z.
        cache : Returns Z as well, useful during backpropagation.
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Parameters:
    -----------
        dA : Post-activation gradient, of any shape.
        cache : 'Z' variable for computing backward propagation efficiently.

    Returns:
    --------
        dZ : Gradient of the cost with respect to Z.
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Parameters:
    -----------
        dA : Post-activation gradient, of any shape.
        cache : 'Z' variable for computing backward propagation efficiently.

    Returns:
    --------
        dZ : Gradient of the cost with respect to Z.
    """
    
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def initialize_parameters(layer_dims):
    """
    Parameters:
    -----------
        layer_dims : Python array (list) containing the dimensions of each layer in the network.
    
    Returns:
    --------
        parameters : Python dictionary containing the parameters "W1", "b1", ..., "WL", "bL":
                         Wl : weight matrix, of shape (layer_dims[l], layer_dims[l-1])
                         bl : bias vector, of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) 

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Parameters:
    -----------
        A : Activations from previous layer (or input data), of shape (size of previous layer, number of examples).
        W : Weights matrix: numpy array, of shape (size of current layer, size of previous layer).
        b : Bias vector, numpy array, of shape (size of the current layer, 1).

    Returns:
    --------
        Z : The input of the activation function, also called pre-activation parameter.
        cache : A python dictionary containing "A", "W" and "b"; 
                stored for computing the backward pass efficiently.
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer.

    Parameters:
    -----------
        A_prev : Activations from previous layer (or input data), of shape (size of previous layer, number of examples).
        W : Weights matrix: numpy array, of shape (size of current layer, size of previous layer).
        b : Bias vector, numpy array, of shape (size of the current layer, 1).
        activation : The activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

    Returns:
    --------
        A : The output of the activation function, also called the post-activation value.
        cache : A python dictionary containing "linear_cache" and "activation_cache";
                stored for computing the backward pass efficiently.
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
    
    Parameters:
    -----------
        X : Input data, of shape (input size, number of examples).
        parameters : Output of initialize_parameters().
    
    Returns:
    --------
        AL : Last post-activation value.
        caches : List of caches containing:
                 every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2),
                 the cache of linear_sigmoid_forward() (there is one, indexed L-1).
    """

    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y, epsilon):
    """
    Implement the cost function.

    Parameters:
    -----------
        AL : Probability vector corresponding to the label predictions, of shape (1, number of examples).
        Y : True "label" vector, of shape (1, number of examples).
        epsilon : Hyperparameter preventing division by zero.

    Returns:
    --------
        cost : Cross-entropy cost.
    """
           
    m = Y.shape[1]
    
    cost = (1./m) * (-np.dot(Y,np.log(AL+epsilon).T) - np.dot(1-Y, np.log(1-AL+epsilon).T))
      
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost


def compute_cost_with_regularization(AL, Y, epsilon, parameters, lambd):
    """
    Implement the cost function with L2 regularization.
    
    Parameters:
    -----------
        AL : post-activation, output of forward propagation, of shape (output size, number of examples)
        Y : True "label" vector, of shape (1, number of examples).
        epsilon : Hyperparameter preventing division by zero.
        parameters : python dictionary containing parameters of the model.
        lambd : Regularization hyperparameter, scalar.
    
    Returns:
    --------
        cost : Value of the regularized loss function.
    """
    m = Y.shape[1]
    W_values = [value for key,value in parameters.items() if "W" in key]
    
    cross_entropy_cost = compute_cost(AL, Y, epsilon) # This gives the cross-entropy part of the cost
    L2_regularization_cost = np.sum([np.sum(np.square(Wl)) for Wl in W_values]) * (1/m) * (lambd/2)
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost


def linear_backward(dZ, cache, lambd):
    """
    Implement the linear portion of backward propagation for a single layer (layer l).

    Parameters:
    -----------
        dZ : Gradient of the cost with respect to the linear output (of current layer l).
        cache : Tuple of values (A_prev, W, b) coming from the forward propagation in the current layer.
        lambd : Regularization hyperparameter, scalar.

    Returns:
    --------
        dA_prev : Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev.
        dW : Gradient of the cost with respect to W (current layer l), same shape as W.
        db : Gradient of the cost with respect to b (current layer l), same shape as b.
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T) + (lambd * W / m)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Parameters:
    -----------
        dA : Post-activation gradient for current layer l.
        cache : Tuple of values (linear_cache, activation_cache) stored for computing backward propagation efficiently.
        activation : The activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
        lambd : Regularization hyperparameter, scalar.
    
    Returns:
    --------
        dA_prev : Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev.
        dW : Gradient of the cost with respect to W (current layer l), same shape as W.
        db : Gradient of the cost with respect to b (current layer l), same shape as b.
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches, epsilon, lambd):
    """
    Implement the backward propagation for the [LINEAR->RELU]*(L-1) -> [LINEAR->SIGMOID] group.
    
    Parameters:
    -----------
        AL : Probability vector, output of the forward propagation (forward_propagation()).
        Y : True "label" vector.
        caches : List of caches containing:
                     every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2).
                     the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1).
        lambd : Regularization hyperparameter, scalar.
    
    Returns:
    --------
        grads : A dictionary with the gradients:
                    grads["dA" + str(l)] = ... 
                    grads["dW" + str(l)] = ...
                    grads["db" + str(l)] = ... 
    """

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL+epsilon) - np.divide(1 - Y, (1 - AL) + epsilon))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid", lambd = lambd)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu",  lambd = lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters_with_gradient_descent(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.
    
    Parameters:
    -----------
        parameters : Python dictionary containing the parameters.
        grads : Python dictionary containing the gradients, output of backward_propagation.
        learning_rate : The learning rate, scalar.
    
    Returns:
    --------
        parameters : Python dictionary containing the updated parameters:
                         parameters["W" + str(l)] = ... 
                         parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
        - keys: "dW1", "db1", ..., "dWL", "dbL".
        - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
    Parameters:
    -----------
        parameters : Python dictionary containing the parameters:
                         parameters['W' + str(l)] = Wl
                         parameters['b' + str(l)] = bl
    
    Returns:
    --------
        v : Python dictionary containing the current velocity:
                v['dW' + str(l)] = velocity of dWl
                v['db' + str(l)] = velocity of dbl
    """
    
    L = len(parameters) // 2
    v = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum.
    
    Parameters:
    -----------
        parameters : Python dictionary containing the parameters:
                         parameters['W' + str(l)] = Wl
                         parameters['b' + str(l)] = bl
        grads : Python dictionary containing the gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
        v : Python dictionary containing the current velocity:
                v['dW' + str(l)] = ...
                v['db' + str(l)] = ...
        beta : The momentum hyperparameter, scalar.
        learning_rate : The learning rate, scalar.
    
    Returns:
    --------
        parameters : Python dictionary containing the updated parameters.
        v : Python dictionary containing the updated velocities.
    """

    L = len(parameters) // 2
    
    for l in range(L):
        
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads["db" + str(l+1)]

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        
    return parameters, v


def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
        - keys: "dW1", "db1", ..., "dWL", "dbL".
        - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Parameters:
    -----------
        parameters : Python dictionary containing the parameters.
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl
    
    Returns:
    --------
        v : Python dictionary that will contain the exponentially weighted average of the gradient:
                v["dW" + str(l)] = ...
                v["db" + str(l)] = ...
        s : Python dictionary that will contain the exponentially weighted average of the squared gradient:
                s["dW" + str(l)] = ...
                s["db" + str(l)] = ...

    """
    
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam.
    
    Parameters:
    -----------
        parameters : Python dictionary containing the parameters:
                         parameters['W' + str(l)] = Wl
                         parameters['b' + str(l)] = bl
        grads : Python dictionary containing the gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
        v : Adam variable, moving average of the first gradient, python dictionary.
        s : Adam variable, moving average of the squared gradient, python dictionary.
        t : Adam counter.
        learning_rate : The learning rate, scalar.
        beta1 : Exponential decay hyperparameter for the first moment estimates.
        beta2 : Exponential decay hyperparameter for the second moment estimates.
        epsilon : Hyperparameter preventing division by zero.

    Returns:
    --------
        parameters : Python dictionary containing the updated parameters.
        v : Adam variable, moving average of the first gradient, python dictionary.
        s : Adam variable, moving average of the squared gradient, python dictionary.
    """

    L = len(parameters) // 2  
    v_corrected = {}       
    s_corrected = {}                         
    
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1-beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1-beta1) * grads["db" + str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l+1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1-beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1-beta2) * np.power(grads['db' + str(l + 1)], 2)

        s_corrected["dW" + str(l+1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l+1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

        if np.any(((np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))) == 0):
            print('found it')
        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - (learning_rate * v_corrected["dW" + str(l + 1)]) / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - (learning_rate * v_corrected["db" + str(l + 1)]) / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)
           
    return parameters, v, s


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random mini-batches from (X, Y).
    
    Parameters:
    -----------
        X : Input data, of shape (input size, number of examples).
        Y : True "label" vector, of shape (1, number of examples).
        mini_batch_size : Size of the mini-batches, integer.

    Returns:
    --------
        mini_batches : List of synchronous (mini_batch_X, mini_batch_Y).
    """
    
    np.random.seed(seed)         
    m = X.shape[1]                
    mini_batches = []
        
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : ]        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def Neural_Net_model(X, Y, layers_dims, optimizer, learning_rate, lambd, batch_type, mini_batch_size, beta,
                     beta1, beta2,  epsilon, num_epochs, print_cost, plot_cost):
    """ 
    Parameters:
    -----------
        X : Input data, of shape (number of features, number of examples).
        Y : True "label" vector, of shape (1, number of examples).
        layers_dims : Python list, containing the size of each layer.
        optimizer: "gradient_descent", "momentum" or "adam".
        learning_rate : The learning rate, scalar.
        lambd : Regularization hyperparameter, scalar.
        batch_type : "batch", "mini_batch" or "stochastic".
        mini_batch_size : The size of a mini batch.
        beta : Momentum hyperparameter.
        beta1 : Exponential decay hyperparameter for the past gradients' estimates. 
        beta2 : Exponential decay hyperparameter for the past squared gradients' estimates.
        epsilon : Hyperparameter preventing division by zero.
        num_epochs : Number of epochs.
        print_cost : True to print the cost every 1000 epochs.
        plot_cost : True to plot a graph of the costs in the different epochs.

    Returns:
    --------
        parameters : Python dictionary containing the updated parameters.
    """
    
    m = X.shape[1]
    L = len(layers_dims)          
    costs = []                     
    t = 0                           
    seed = 10                       
    
    parameters = initialize_parameters(layers_dims)
    
    assert optimizer in ["gradient_descent", "momentum", "adam"], "invalid optimizer"

    if optimizer == "gradient_descent":
        pass 
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
        
    assert batch_type in ["batch", "mini_batch", "stochastic"], "invalid batch type"
    
    if batch_type == "batch":
        mini_batch_size = m
    elif batch_type == "stochastic":
        mini_batch_size = 1
    elif batch_type == "mini_batch":
        pass
    
    for i in range(num_epochs):

        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch

            AL, caches = forward_propagation(minibatch_X, parameters)

            cost = compute_cost_with_regularization(AL, minibatch_Y, epsilon, parameters, lambd)

            grads = backward_propagation(AL, minibatch_Y, caches, epsilon, lambd)

            if optimizer == "gradient_descent":
                parameters = update_parameters_with_gradient_descent(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
 
        
        # Print the cost every 1000 epoch          
        if print_cost and i % 100 == 0:
            print (f"Cost after {i} epochs: {cost}")
        costs.append(cost)
                
    if plot_cost:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters


def predict_yhat(X, parameters):
    """
    This function is used to predict the results of a trained neural network.
    
    Parameters:
    -----------
        X : Input data, of shape (number of features, number of examples).
        parameters : Parameters of the trained model.
    
    Returns:
    --------
        p : Predictions for the given dataset X.
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    probas, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
        
        
    return p


def predict_proba_(X, parameters):
    """
    This function is used to predict the class probabilities of a trained neural network.
    
    Parameters:
    -----------
        X : Input data, of shape (number of features, number of examples).
        parameters : Parameters of the trained model.
    
    Returns:
    --------
        probas : Class probabilities for the given dataset X.
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    probas, caches = forward_propagation(X, parameters)     
        
    return probas


def accuracy_score(y_test, y_pred):
    """
    Parameters:
    ----------
        y_test : True "label" vector, of shape (1, number of examples).
        y_pred : The predicted "label" vector, of shape (1, number of examples).
    
    Returns:
    --------
        score : The accuracy score.
    """
    
    score = np.mean(y_test==y_pred)

    return score


class Neural_Network:
    """
    Parameters:
    -----------
        hidden_layer_sizes : Python list, containing the size of each hidden layer.
        optimizer: "gradient_descent", "momentum" or "adam".
        learning_rate : The learning rate, scalar.
        lambd : Regularization hyperparameter, scalar.
        batch_type : "batch", "mini_batch" or "stochastic".
        mini_batch_size : The size of a mini batch.
        beta : Momentum hyperparameter.
        beta1 : Exponential decay hyperparameter for the past gradients' estimates. 
        beta2 : Exponential decay hyperparameter for the past squared gradients' estimates.
        epsilon : Hyperparameter preventing division by zero.
        num_epochs : Number of epochs.
        print_cost : True to print the cost every 1000 epochs.
        plot_cost : True to plot a graph of the costs in the different epochs.        
     
     Attributes:
     -----------
         is_fitted : True for a fitted model, False for model that hasn't been fitted yet.
         parameters : The parameters of the trained model.
         
     Methods:
     --------
         fit : Train the model using the training set (X, y).
         predict : Predict class value for X.
         predict_proba : Predict class probabilities of the input samples X.
         score : Returns the mean accuracy on the given test data and labels.
     """
    
    def __init__(self, hidden_layer_sizes = [7], optimizer = 'gradient_descent', learning_rate = 0.0075, lambd = 0,
                 batch_type = 'batch', mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, 
                 num_epochs = 10000, print_cost = False, plot_cost = False):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.batch_type = batch_type
        self.mini_batch_size = mini_batch_size
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.print_cost = print_cost
        self.plot_cost = plot_cost
        self.parameters = None
        self.is_fitted = False


    def fit(self, X, y):
        """
        Train the model using the training set (X, y).
        
        Parameters:
        -----------
            X : Input data, of shape (input size, number of examples).
            y : True "label" vector, of shape (1, number of examples).       
        """
        self.layers_dims = [X.shape[0]] + self.hidden_layer_sizes + [1]      
        self.parameters = Neural_Net_model(X, y, self.layers_dims, self.optimizer, self.learning_rate, self.lambd, self.batch_type,
                                           self.mini_batch_size, self.beta, self.beta1, self.beta2, self.epsilon, 
                                           self.num_epochs, self.print_cost, self.plot_cost)
        
        
        self.is_fitted = True
      

    def predict(self, X):
        """
        Predict class value for X.
        
        Parameters:
        -----------
            X : Input data, of shape (input size, number of examples).
            
        Returns:
        --------
            y_hat : Predictions for the given dataset X.
        """
        assert self.is_fitted, "The model is not fitted yet"

        y_hat = predict_yhat(X, self.parameters)
        
        return y_hat
    
    
    def predict_proba(self, X):
        """
        Predict class probabilities of the input dataset X.
        
        Parameters:
        -----------
            X : Input data, of shape (input size, number of examples).
            
        Returns:
        --------
            y_pred_proba : Predict class probabilities of the input dataset X.
        """
        assert self.is_fitted, "The model is not fitted yet"
        
        y_pred_proba = predict_proba_(X, self.parameters)
        
        return y_pred_proba
    
    
    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
            X : Input data, of shape (input size, number of examples).
            y : True "label" vector, of shape (1, number of examples).
            
        Returns:
        --------
            acc_score : Mean accuracy score.
        """
        assert self.is_fitted, "The model is not fitted yet"
        
        y_hat = predict_yhat(X, self.parameters)
        acc_score = accuracy_score(y, y_hat)
        
        return acc_score
