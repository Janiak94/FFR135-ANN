import numpy as np
import matplotlib.pyplot as plt

def normalize_data(data):
    return (data-np.mean(data)) / np.std(data)

def get_random_array(shape):
    """
    returns array with entries
    uniformely distributed between -1 and 1
    and with dim as in shape
    """
    w = 2 * np.random.rand(shape[0], shape[1]) - 1
    return w

def get_network_parameters(layer_sizes):
    """
    Generate weights and tresholds for gives set of neurons
    in each layer
    """
    weights = []
    tresholds = []
    for i in range(len(layer_sizes)-1):
        weights_shape = (layer_sizes[i+1], layer_sizes[i])
        weights.append(get_random_array(weights_shape))
        tresholds.append(np.zeros((weights_shape[0],1)))
    return (weights, tresholds)

def feed_forward(input_data, weights, tresholds):
    """
    Perform one step feed forward
    """
    depth = len(weights)
    state = input_data.reshape((-1, 1)) # Make sure we have colomn vector
    for i in range(depth):
        local_field = weights[i] @ state - tresholds[i]
        state = np.tanh(local_field)
    return np.sign(np.squeeze(state))

def stochastic_gradient_descent(training_data, weights, tresholds, n_iter, learning_rate):
    training_inputs = training_data[:,0:2]
    training_labels = training_data[:,-1]
    activation_function = np.tanh
    activation_derivative = (lambda x: (1 / np.cosh(x))**2)
    depth = len(weights) # Number of hidden layers plus output layer
    for i in range(n_iter):
        r = np.random.randint(training_inputs.shape[0]) # Chose random pattern index
        state = training_inputs[r,:].reshape((-1,1)) # Set initial state (input neurons)
        local_fields = []
        states = [state]
        errors = []
        for i in range(depth): # Perform feed forward
            local_field = weights[i] @ state - tresholds[i]
            state = np.tanh(local_field) # Update state
            local_fields.append(local_field) # Save values for backpropagation later
            states.append(state) # Save values for when updating weights
        output = state
        output_error = activation_derivative(local_fields[-1]) * (training_labels[r] - output)
        errors = [output_error]
        for i in range(depth-1, 0, -1): # Perform backpropagation
            t1 = (errors[0] @ weights[i]).T
            t2 = activation_derivative(local_fields[i-1])
            err = np.multiply(t1, t2)
            errors.insert(0, err)
        for i in range(depth): # Update parameters
            weights[i] = weights[i] + learning_rate * np.outer(errors[i], states[i])
            tresholds[i] = tresholds[i] - learning_rate * errors[i]
    return (weights, tresholds)


############################
# Import data sets
############################
TRAINING_DATA_PATH = "training_data/training_set.csv"
VALIDATION_DATA_PATH = "training_data/validation_set.csv"

training_data = np.genfromtxt(TRAINING_DATA_PATH, delimiter=",")
validation_data = np.genfromtxt(VALIDATION_DATA_PATH, delimiter=",")

training_data = 2*np.random.rand(10000, 3)-1
for i in range(10000):
    training_data[i,2] = 1 if (training_data[i,0]**2 + training_data[i,1]**2 < 0.5) else 0


############################
# Normalize data
############################
validation_data[:,0:2] = normalize_data(validation_data[:,0:2])

#############################
# Number of neurons in each layer
#############################
M1 = 30 # Number of hidden neurons
input_size = training_data.shape[1] - 1 # Subtract 1 for the labels
output_size = 1
layer_sizes = [input_size, M1, output_size]

#############################
# Initialize weights and treshold
#############################
(weights, tresholds) = get_network_parameters(layer_sizes)

pred2 = []
for i in range(training_data.shape[0]):
    output = feed_forward(training_data[i,0:2], weights, tresholds)
    pred2.append(output)
print(weights)
print(tresholds)
#############################
# Run stochastic gradient descent
#############################
(weights, tresholds) = stochastic_gradient_descent(training_data, weights, tresholds, 1000000, 0.0001)
print(weights)
print(tresholds)
#############################
# Predict
#############################
pred1 = []
for i in range(training_data.shape[0]):
    output = feed_forward(training_data[i,0:2], weights, tresholds)
    pred1.append(output)

#############################
# Plot training data and predicted data
#############################
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.scatter(training_data[:,0], training_data[:,1], c=training_data[:,-1], s=1)
ax2.scatter(training_data[:,0], training_data[:,1], c=pred1, s=1)
ax3.scatter(training_data[:,0], training_data[:,1], c=pred2, s=1)
plt.show()
plt.plot()