#coding=utf-8
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numpy import *


#栈式编码代价计算
def stacked_ae_cost(theta, input_size, hidden_size,
                    n_classes, net_config, lambda_, data, labels):

    # We first extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size*n_classes].reshape((n_classes, hidden_size))

    # Extract out the "stack"
    stack = params2stack(theta[hidden_size*n_classes:], net_config)

    # Number of examples
    m = data.shape[1]

    # Forword pass
    z = [np.zeros(1)] # Note that z[0] is dummy
    a = [data]
    for s in stack:
        z.append(s['w'].dot(a[-1]) + s['b'].reshape((-1, 1)) )
        a.append(sigmoid(z[-1]))

    learned_features = a[-1]

    # Probability with shape (n_classes, m)
    theta_features = softmax_theta.dot(learned_features)
    alpha = np.max(theta_features, axis=0)
    theta_features -= alpha # Avoid numerical problem due to large values of exp(theta_features)
    proba = np.exp(theta_features) / np.sum(np.exp(theta_features), axis=0)

    # Matrix of indicator fuction with shape (n_classes, m)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.arange(m))))
    indicator = np.array(indicator.todense())

    # Compute softmax cost and gradient
    cost = -1.0/m * np.sum(indicator * np.log(proba)) + 0.5*lambda_*np.sum(softmax_theta*softmax_theta)
    softmax_grad = -1.0/m * (indicator - proba).dot(learned_features.T) + lambda_*softmax_theta

    # Backpropagation
    delta = [- softmax_theta.T.dot(indicator - proba) * sigmoid_prime(z[-1])]
    n_stack = len(stack)
    for i in reversed(range(n_stack)): # Note that delta[0] will not be used
        d = stack[i]['w'].T.dot(delta[0])*sigmoid_prime(z[i])
        delta.insert(0, d) # Insert element at beginning

    stack_grad = [{} for i in range(n_stack)]
    for i in range(n_stack):
        stack_grad[i]['w'] = delta[i+1].dot(a[i].T) / m
        stack_grad[i]['b'] = np.mean(delta[i+1], axis=1)

    stack_grad_params = stack2params(stack_grad)[0]

    grad = np.concatenate((softmax_grad.flatten(), stack_grad_params))

    return cost, grad


#栈式编码预测
def stacked_ae_predict(theta, input_size, hidden_size,
                       n_classes, net_config, data):

    # We first extract the part which compute the softmax gradient
    softmax_theta = theta[0:hidden_size*n_classes].reshape((n_classes, hidden_size))

    # Extract out the "stack"
    stack = params2stack(theta[hidden_size*n_classes:], net_config)

    # Number of examples
    m = data.shape[1]

    # Forword pass
    z = [np.zeros(1)]
    a = [data]
    for s in stack:
        z.append(s['w'].dot(a[-1]) + s['b'].reshape((-1, 1)) )
        a.append(sigmoid(z[-1]))

    learned_features = a[-1]

    # Softmax model
    model = {}
    model['opt_theta']  = softmax_theta
    model['n_classes']  = n_classes
    model['input_size'] = hidden_size

    # Make predictions
    pred = softmax_predict(model, learned_features)

    return pred


#栈式编码梯度校验
def check_stacked_ae_cost():
    # Setup random data / small model
    input_size = 4;
    hidden_size = 5;
    lambda_ = 0.01;
    data   = np.random.randn(input_size, 5)
    labels = np.array([ 0, 1, 0, 1, 0], dtype=np.uint8)
    n_classes = 2
    n_stack = 2

    stack = [{} for i in range(n_stack)]
    stack[0]['w'] = 0.1 * np.random.randn(3, input_size)
    stack[0]['b'] = np.zeros(3)
    stack[1]['w'] = 0.1 * np.random.randn(hidden_size, 3)
    stack[1]['b'] = np.zeros(hidden_size)

    softmax_theta = 0.005 * np.random.randn(hidden_size * n_classes)

    stack_params, net_config = stack2params(stack)
    stacked_ae_theta = np.concatenate((softmax_theta, stack_params))

    cost, grad = stacked_ae_cost(stacked_ae_theta, input_size, hidden_size,
                                 n_classes, net_config, lambda_, data, labels)

    # Check that the numerical and analytic gradients are the same
    J = lambda theta : stacked_ae_cost(theta, input_size, hidden_size,
                                       n_classes, net_config, lambda_, data, labels)[0]
    nume_grad = compute_numerical_gradient(J, stacked_ae_theta)

    # Use this to visually compare the gradients side by side
    for i in range(grad.size):
        print("{0:20.12f} {1:20.12f}".format(nume_grad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Compare numerically computed gradients with the ones obtained from backpropagation
    # The difference should be small. In our implementation, these values are usually less than 1e-9.
    # When you got this working, Congratulations!!!
    diff = np.linalg.norm(nume_grad - grad) / np.linalg.norm(nume_grad + grad)
    print("Norm of difference = ", diff)
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')



#栈式编码器参数设置
def stack2params(stack):

    # Setup the compressed param vector
    params = []
    for i in range(len(stack)):
        w = stack[i]['w']
        b = stack[i]['b']
        params.append(w.flatten())
        params.append(b.flatten())

        # Check that stack is of the correct form
        assert w.shape[0] == b.size, \
            'The size of bias should equals to the column size of W for layer {}'.format(i)
        if i < len(stack)-1:
            assert stack[i]['w'].shape[0] == stack[i+1]['w'].shape[1], \
                'The adjacent layers L {} and L {} should have matching sizes.'.format(i, i+1)

    params = np.concatenate(params)

    # Setup network configuration
    net_config = {}
    if len(stack) == 0:
        net_config['input_size'] = 0
        net_config['layer_sizes'] = []
    else:
        net_config['input_size'] = stack[0]['w'].shape[1]
        net_config['layer_sizes'] = []
        for s in stack:
            net_config['layer_sizes'].append(s['w'].shape[0])

    return params, net_config


def params2stack(params, net_config):
    """
    Converts a flattened parameter vector into a nice "stack" structure
    for us to work with. This is useful when you're building multilayer
    networks.

    params: flattened parameter vector
    net_config: auxiliary variable containing the configuration of the network
    """

    # Map the params (a vector into a stack of weights)
    layer_sizes = net_config['layer_sizes']
    prev_layer_size = net_config['input_size'] # the size of the previous layer
    depth = len(layer_sizes)
    stack = [{} for i in range(depth)]
    current_pos = 0                           # mark current position in parameter vector

    for i in range(depth):
        # Extract weights
        wlen = layer_sizes[i] * prev_layer_size
        stack[i]['w'] = params[current_pos:current_pos+wlen].reshape((layer_sizes[i], prev_layer_size))
        current_pos += wlen

        # Extract bias
        blen = layer_sizes[i]
        stack[i]['b'] = params[current_pos:current_pos+blen]
        current_pos += blen

        # Set previous layer size
        prev_layer_size = layer_sizes[i]

    return stack





def compute_numerical_gradient(J, theta):
    """
    J: a function that outputs a real-number. Calling y = J(theta) will return the
       function value at theta. 
    theta: a vector of parameters
    """
    n = theta.size
    grad = np.zeros(n)
    eps = 1.0e-4
    eps2 = 2*eps
    
    for i in range(n):
        theta_p = theta.copy()
        theta_n = theta.copy()
        theta_p[i] = theta[i] + eps
        theta_n[i] = theta[i] - eps
        
        grad[i] = (J(theta_p) - J(theta_n)) / eps2
    
    return grad

#计算softmax代价
def softmax_cost(theta, n_classes, input_size, lambda_, data, labels):
    k = n_classes
    n, m = data.shape
    theta = theta.reshape((k, n))
    theta_data = theta.dot(data)
    alpha = np.max(theta_data, axis=0)
    theta_data -= alpha 
    proba = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.arange(m))))
    indicator = np.array(indicator.todense())
    cost = -1.0/m * np.sum(indicator * np.log(proba)) + 0.5*lambda_*np.sum(theta*theta)
    grad = -1.0/m * (indicator - proba).dot(data.T) + lambda_*theta
    grad = grad.ravel()

    return cost, grad



def softmax_predict(model, data):

    theta = model['opt_theta'] # Optimal theta
    k = model['n_classes']  # Number of classes
    n = model['input_size'] # Input size (number of features)

    # Reshape theta
    theta = theta.reshape((k, n))

    # Probability with shape (k, m)
    theta_data = theta.dot(data)
    alpha = np.max(theta_data, axis=0)
    theta_data -= alpha # Avoid numerical problem due to large values of exp(theta_data)
    proba = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)

    # Prediction values
    pred = np.argmax(proba, axis=0)

    return pred



def display_network(A):
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = np.ceil(np.sqrt(col))
    m = np.ceil(col / n)

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    return image



#稀疏编码器代价计算
def sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    # Number of instances
    m = data.shape[1]

    # Forward pass
    a1 = data              # Input activation
    z2 = W1.dot(a1) + b1.reshape((-1, 1))
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2.reshape((-1, 1))
    h  = sigmoid(z3)       # Output activation
    y  = a1

    # Compute rho_hat used in sparsity penalty
    rho = sparsity_param
    rho_hat = np.mean(a2, axis=1)
    sparsity_delta = (-rho/rho_hat + (1.0-rho)/(1-rho_hat)).reshape((-1, 1))

    # Backpropagation
    delta3 = (h-y)*sigmoid_prime(z3)
    delta2 = (W2.T.dot(delta3) + beta*sparsity_delta)*sigmoid_prime(z2)

    # Compute the cost
    squared_error_term = np.sum((h-y)**2) / (2.0*m)
    weight_decay = 0.5*lambda_*(np.sum(W1*W1) + np.sum(W2*W2))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))
    cost = squared_error_term + weight_decay + sparsity_term

    # Compute the gradients
    W1grad = delta2.dot(a1.T)/m + lambda_*W1
    W2grad = delta3.dot(a2.T)/m + lambda_*W2
    b1grad = np.mean(delta2, axis=1)
    b2grad = np.mean(delta3, axis=1)
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad


#前向传播自动编码
def feedforward_autoencoder(theta, hidden_size, visible_size, data):
   
    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size].reshape((-1, 1))
    
    # Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.
    a1 = data
    z2 = W1.dot(a1) + b1
    a2 = sigmoid(z2)

    return a2


def sigmoid(inX):
    return 1.0/(1+exp(-inX)) 


def sigmoid_prime(x):
    f = sigmoid(x)
    df = f*(1.0-f)
    return df

def initialize_parameters(hidden_size, visible_size):
    # Initialize parameters randomly based on layer sizes.
    r  = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    # we'll choose weights uniformly from the interval [-r, r)
    W1 = np.random.random((hidden_size, visible_size)) * 2.0 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2.0 * r - r

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)

    theta = np.hstack((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))

    return theta


def KL_divergence(p, q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))



#softmax训练
def softmax_train(input_size, n_classes, lambda_, input_data, labels, options={'maxiter': 400, 'disp': True}):
    # initialize parameters
    theta = 0.005 * np.random.randn(n_classes * input_size)

    J = lambda theta : softmax_cost(theta, n_classes, input_size, lambda_, input_data, labels)

    # Find out the optimal theta
    results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
    opt_theta = results['x']

    model = {'opt_theta': opt_theta, 'n_classes': n_classes, 'input_size': input_size}
    return model

def load_MNIST_images(filename):	#加载
    with open(filename, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        n_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((n_images, rows * cols))
        images = images.T
        images = images.astype(np.float64) / 255
        f.close()
        return images


def load_MNIST_labels(filename):	#return labels
    with open(filename, 'r') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        n_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        labels = np.fromfile(f, dtype=np.uint8)
        f.close()
        return labels



input_size = 28 * 28
n_classes = 10         # Number of classes
hidden_size_L1 = 200   # Layer 1 Hidden Size
hidden_size_L2 = 200   # Layer 2 Hidden Size
sparsity_param = 0.1   # desired average activation of the hidden units.
                       # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                       #  in the lecture notes).
lambda_ = 3e-3         # weight decay parameter
beta = 3               # weight of sparsity penalty term

maxiter = 400          # Maximum iterations for training

"""
STEP 1: Load data from the MNIST database

  This loads our training data from the MNIST database files.
"""

# Load MNIST database files
# Load MNIST database files
train_data   = load_MNIST_images('data/mnist/train-images-idx3-ubyte')
train_labels = load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')



#训练第一层稀疏编码器
# Randomly initialize the parameters
sae1_theta = initialize_parameters(hidden_size_L1, input_size)

#  Instructions: Train the first layer sparse autoencoder, this layer has
#                an hidden size of "hidden_size_L1"
#                You should store the optimal parameters in sae1_opt_theta

J = lambda theta : sparse_autoencoder_cost(theta, input_size, hidden_size_L1, lambda_, sparsity_param, beta, train_data)

options = {'maxiter': maxiter, 'disp': True}

results = scipy.optimize.minimize(J, sae1_theta, method='L-BFGS-B', jac=True, options=options)
sae1_opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)

# Visualize weights
visualize_weights = False
if visualize_weights:
    W1 = sae1_opt_theta[0:hidden_size_L1*input_size].reshape((hidden_size_L1, input_size))
    image = display_network(W1.T)
    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()



#训练第二层稀疏编码器
sae1_features = feedforward_autoencoder(sae1_opt_theta, hidden_size_L1, input_size, train_data)

#  Randomly initialize the parameters
sae2_theta = initialize_parameters(hidden_size_L2, hidden_size_L1)

#  Instructions: Train the second layer sparse autoencoder, this layer has
#                an hidden size of "hidden_size_L2" and an input size of "hidden_size_L1"
#                You should store the optimal parameters in sae2_opt_theta
J = lambda theta : sparse_autoencoder_cost(theta, hidden_size_L1, hidden_size_L2,
    lambda_, sparsity_param, beta, sae1_features)

options = {'maxiter': maxiter, 'disp': True}

results = scipy.optimize.minimize(J, sae2_theta, method='L-BFGS-B', jac=True, options=options)
sae2_opt_theta = results['x']

print("Show the results of optimization as following.\n")
print(results)



#训练softmax
sae2_features = feedforward_autoencoder(sae2_opt_theta, hidden_size_L2, hidden_size_L1, sae1_features)


options = {'maxiter': maxiter, 'disp': True}
softmax_model = softmax_train(hidden_size_L2, n_classes, lambda_, sae2_features, train_labels, options)
softmax_opt_theta = softmax_model['opt_theta']


#微调
n_stack = 2 # Two layers
stack = [{} for i in range(n_stack)]

stack[0]['w'] = sae1_opt_theta[0:hidden_size_L1*input_size].reshape((hidden_size_L1, input_size))
stack[0]['b'] = sae1_opt_theta[2*hidden_size_L1*input_size: 2*hidden_size_L1*input_size + hidden_size_L1]

stack[1]['w'] = sae2_opt_theta[0:hidden_size_L2*hidden_size_L1].reshape((hidden_size_L2, hidden_size_L1))
stack[1]['b'] = sae2_opt_theta[2*hidden_size_L2*hidden_size_L1: 2*hidden_size_L2*hidden_size_L1 + hidden_size_L2]

# Initialize the parameters for the deep model
stack_params, net_config = stack2params(stack)
stacked_ae_theta = np.concatenate((softmax_opt_theta, stack_params))

# Instructions: Train the deep network, hidden size here refers to the
#               dimension of the input to the classifier, which corresponds
#               to "hidden_size_L2".

J = lambda theta : stacked_ae_cost(theta, input_size, hidden_size_L2, n_classes, net_config, lambda_, train_data, train_labels)

#check_stacked_ae_cost() # Verify the correctness

# Find out the optimal theta
options = {'maxiter': maxiter, 'disp': True}
results = scipy.optimize.minimize(J, stacked_ae_theta, method='L-BFGS-B', jac=True, options=options)
stacked_ae_opt_theta = results['x']

print(results)

#测试
test_data   = load_MNIST_images('data/mnist/t10k-images-idx3-ubyte')
test_labels = load_MNIST_labels('data/mnist/t10k-labels-idx1-ubyte')

pred = stacked_ae_predict(stacked_ae_theta, input_size, hidden_size_L2, n_classes, net_config, test_data)

acc = np.mean(test_labels == pred)
print("Before Finetuning Test Accuracy: {:5.2f}% \n".format(acc*100))

pred = stacked_ae_predict(stacked_ae_opt_theta, input_size, hidden_size_L2, n_classes, net_config, test_data)

acc = np.mean(test_labels == pred)
print("After Finetuning Test Accuracy: {:5.2f}% \n".format(acc*100))

