## Tutor: Vassilis Diakoloukas 
## Year: 2022

import numpy as np
from keras.datasets import mnist
from keras import utils
import matplotlib.pyplot as plt

# Neural Network Dense (Fully Connected) Layer without activation
class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    #Forward Propagation on a Dense Layer
    def forward(self, input):
        self.input = input
        # Add Code Here
        fwd = np.dot(self.weights, input) + self.bias
        return fwd

    #Backward Propagation on a Dense Layer
    # dE_dY is dE/dY Gradient
    # dE_dW is dE/dW Gradient
    # dE_dB is dE/dB Gradient
    # dE_dX is dE/dX Gradient
    def backward(self, dE_dY, learning_rate):
        # Add Code Here
        dE_dW = np.dot(dE_dY, self.input.T)
        dE_dX = np.dot(self.weights.T, dE_dY)
        dE_dB = dE_dY

        self.update_weights(dE_dW, dE_dB, learning_rate)
        return dE_dX

    # Update Layer Weights and bias
    def update_weights(self, dE_dW, dE_dB, learning_rate):
        # Add Code Here
        self.weights = self.weights - learning_rate*dE_dW
        self.bias = self.bias - learning_rate*dE_dB


# Neural Network Activation Layer Abstract Class
# Consider Activation as a separate layer for more flexibility
# Properties and methods will be inherited into the specific activation function classes
class Activation():
    def __init__(self, activation, activation_grad):
        self.activation = activation
        self.activation_grad = activation_grad

    # input: is the input to the activation function
    # Y: is the output of the activation function
    def forward(self, input):
        self.input = input
        # Add Code Here
        Y = self.activation(self.input)
        return Y

    # Backward estimation of dE/dX using the activation prime (derivative)
    def backward(self, dE_dY, learning_rate):
        # Add Code Here
        dE_dX = dE_dY*self.activation_grad(self.input)
        return dE_dX
 

# Softmax Activation Function
# Should be used in the output layer especially when Cross-Entropy is considered
class Softmax():
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, grad):  #????????????? grad
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


# Tanh Activation Function
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            act = np.tanh(x)
            return act

        def tanh_grad(x):
            # Add Code Here
            actGrad = 1 - (np.tanh(x))**2
            return actGrad

        super().__init__(tanh, tanh_grad)


# Sigmoid (Logistic) Activation Function
class Sigmoid(Activation):
    def __init__(self):
        # Logistic Activation Function
        def sigmoid(x):
            # Add Code Here
            act = 1/(1+np.exp(-x))   ##?????? (-x)
            return act

        # Activation Function Gradient (Derivative)
        def sigmoid_grad(x):
            # Add Code Here
            temp = 1 / (1 + np.exp(-x))
            actGrad = (1-temp)*temp
            #actGrad = np.exp(-x)/(1+np.exp(-x))
            return actGrad

        super().__init__(sigmoid, sigmoid_grad)

        
# Return the the cross entropy loss
def loss_cross_entropy(y_true, y_pred):
    # Add Code Here
    loss = np.mean(-y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred))
    return loss

# Return the derivative of the cross entropy loss
def loss_cross_entropy_grad(y_true, y_pred):
    # Add Code Here
    lossGrad = (y_pred-y_true) #/ (y_pred * (1-y_pred))
    return lossGrad

def mse(y_true, y_pred):
    # Add Code Here
    loss = np.mean(np.power(y_pred - y_true, 2))
    return loss

def mse_grad(y_true, y_pred):
    # Add Code Here
    lossGrad = (2/y_true.size)*(y_pred - y_true)
    return lossGrad

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def reshuffle(X, Y):
    NData = len(X)
    perm_indices = np.arange(NData)
    np.random.shuffle(perm_indices)
    X = X[perm_indices]
    Y = Y[perm_indices]
    return X, Y

def read_next_batch(X, Y, batch_size, batch_idx=0):
    NData = len(X)
    if batch_idx + batch_size < NData:
        X_batch = X[batch_idx:batch_idx+batch_size]
        Y_batch = Y[batch_idx:batch_idx+batch_size]
        batch_idx = batch_idx + batch_size
        return X_batch, Y_batch, batch_idx
    else:
        return None, None, None

def train(network, x_train, y_train, epochs = 1000, learning_rate = 0.01, batch_size = 128, verbose = True):
    num_batches = 0
    for epoch in range(epochs):
        epoch_loss = 0
        x_train, y_train = reshuffle(x_train, y_train)
        batch_idx = 0
        x_batch, y_batch, batch_idx = read_next_batch(x_train, y_train, batch_size, batch_idx)
        while x_batch is not None:
            num_batches += 1
            for x, y in zip(x_batch, y_batch):
                # forward pass
                output = predict(network, x)
                # Epoch Loss/Error based on prediction
                # Use cross_entropy
                epoch_loss += loss_cross_entropy(y, output)
                # Use MSE
                #epoch_loss += mse(y, output)

                # backward Error Propagation
                grad = loss_cross_entropy_grad(y, output)

                #grad = mse_grad(y, output)

                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

            x_batch, y_batch, batch_idx = read_next_batch(x_train, y_train, batch_size, batch_idx)

            #epoch_loss /= len(x_train)
            epoch_loss /= num_batches
        if verbose:
            print(f"{epoch + 1}/{epochs}, error={epoch_loss}")

##################################################################
###### Start Running the code from here

# load MNIST using Keras
# Select 1000 training samples and 20 test samples and appropriate preprocess them
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# Build the Neural Network Architecture
# Change the layer name and parameters appropriately to experiment with
network = [
    Dense(28 * 28, 100),
    Dense(100, 20),
    #Dense(28 * 28, 10),
    Sigmoid(),
    #Tanh(),
    Dense(20, 10),
    Softmax()
]

# train the network using the input data and stochastic Gradient Descent
# Define different learning rates, epochs and batch_size to experiment with
train(network, x_train, y_train, epochs=100, learning_rate=0.1, batch_size = 32)

# Evaluate performance on test data
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

ratio = sum([np.argmax(y) == np.argmax(predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
toterror = sum([mse(y, predict(network, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print('ratio: %.2f' % ratio)
print('mse: %.4f' % toterror)


# Plot 10 samples with their corresponding network prediction and true label
samples = 10
for test, true in zip(x_test[:samples], y_test[:samples]):
    image = np.reshape(test, (28, 28))
    plt.imshow(image, cmap='binary')
    pred = predict(network, test)
    idx = np.argmax(pred)
    idx_true = np.argmax(true)
    plt.title('pred: %s, prob: %.2f, true: %d' % (idx, pred[idx], idx_true))
    plt.show()
    #print('pred: %s, prob: %.2f, true: %d' % (idx, pred[idx], idx_true))