import numpy as np
import matplotlib.pyplot as plt

def get_random_array(shape):
    """
    returns array with entries
    uniformely distributed between -1 and 1
    and with dim as in shape
    """
    w = 2 * np.random.rand(shape[0], shape[1]) - 1
    return w

def batch_data(training_data, batch_size):
    """
    Form random batches from the training data
    and return them as a list, each entry containing a batch
    """
    n_data_points = training_data.shape[0]
    shuffled_data = training_data.copy() # Make copy so we do not touch the original
    np.random.shuffle(shuffled_data) # Shuffle the data to get different batches in each run
    batches = []
    for i in range(0, n_data_points, batch_size): # Form batches
        batch = shuffled_data[i:(i+batch_size),:]
        batches.append(batch)
    return batches

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

def stochastic_gradient_descent(training_data, weights, tresholds, n_epoch, learning_rate, batch_size):
    activation_function = np.tanh
    activation_derivative = (lambda x: (1 / np.cosh(x))**2)
    depth = len(weights) # Number of hidden layers plus output layer
    for i in range(n_epoch):
        print("Epoch {}/{}".format(i, n_epoch))
        batches = batch_data(training_data, batch_size) # Generate batches of data

        # Perform batch training
        for batch in batches:
            batch_pattern = batch[:,0:2].T # Separate the patterns from labels and make column vectors
            batch_labels = batch[:,-1]
            state = batch_pattern # Set initial state
            local_fields = [] # Store locals fields for each layer and pattern
            states = [state] # Store neuron states for each layer and pattern
            errors = [] # Store error terms for each layer and pattern

            # Perform feed forward
            for j in range(depth):
                local_field = weights[j] @ state - tresholds[j] # Should return array with shape == (x, batch_pattern.shape[1])
                state = activation_function(local_field) # Shouldnt change the shape
                local_fields.append(local_field) # Save field for later
                states.append(state) # Save state for later
            output = state
            output_error = activation_derivative(local_fields[-1]) * (batch_labels - output) # Calculate the output errors for each pattern

            errors = [output_error]
            # Perform backpropagation
            for j in range(depth-1, 0, -1):
                t1 = errors[0].T @ weights[j]
                t2 = activation_derivative(local_fields[j-1])
                err = t1.T * t2
                errors.insert(0, err)

            # Update weights
            for j in range(depth):
                dw = np.zeros(weights[j].shape)
                dt = np.zeros(tresholds[j].shape).T
                # Add errors for each pattern (use np sum)
                for k in range(batch_pattern.shape[1]):
                    dw += np.outer(errors[j][:,k], states[j][:,k])
                    dt += errors[j][:,k]
                weights[j] += learning_rate * dw
                tresholds[j] -= learning_rate * dt.T
    return (weights, tresholds)

if __name__ == "__main__":
    ############################
    # Import data sets
    ############################
    TRAINING_DATA_PATH = "training_data/training_set.csv"
    VALIDATION_DATA_PATH = "training_data/validation_set.csv"

    training_data = np.genfromtxt(TRAINING_DATA_PATH, delimiter=",")
    validation_data = np.genfromtxt(VALIDATION_DATA_PATH, delimiter=",")

    ############################
    # Normalize data
    ############################
    mean = np.mean(training_data[:,0:2])
    std = np.std(training_data[:,0:2])
    training_data[:,0:2] = (training_data[:,0:2] - mean) / std
    validation_data[:,0:2] = (validation_data[:,0:2] - mean) / std

    #############################
    # Number of neurons in each layer
    #############################
    M1 = 10 # Number of hidden neurons
    input_size = training_data.shape[1] - 1 # Subtract 1 for the labels
    output_size = 1
    layer_sizes = [input_size, M1, output_size]

    #############################
    # Initialize weights and treshold
    #############################
    (weights, tresholds) = get_network_parameters(layer_sizes)

    #############################
    # Run stochastic gradient descent
    #############################
    n_epoch = 5000
    learning_rate = 5e-4
    batch_size = 50
    (weights, tresholds) = stochastic_gradient_descent(
        training_data,
        weights, tresholds,
        n_epoch=n_epoch,
        learning_rate=learning_rate,
        batch_size=batch_size
        )

    #############################
    # Predict
    #############################
    pred = []
    for i in range(training_data.shape[0]):
        output = feed_forward(training_data[i,0:2], weights, tresholds)
        pred.append(output)

    #############################
    # Classification error on validation data set
    #############################
    c_error = 0
    for i in range(validation_data.shape[0]):
        output = feed_forward(validation_data[i,0:2], weights, tresholds)
        c_error += np.abs(output - validation_data[i,2])
    c_error /= 2 * validation_data.shape[0]
    print("Classification error on validation data set: {:.4f}".format(c_error))

    #############################
    # Plot training data and predicted data
    #############################
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(training_data[:,0], training_data[:,1], c=training_data[:,-1], s=1)
    ax2.scatter(training_data[:,0], training_data[:,1], c=pred, s=1)
    plt.show()
    plt.plot()

    #############################
    # Save network paramters
    #############################
    np.savetxt("output/w1.csv", weights[0], delimiter=",")
    np.savetxt("output/w2.csv", weights[1].T, delimiter=",")
    np.savetxt("output/t1.csv", tresholds[0], delimiter=",")
    np.savetxt("output/t2.csv", tresholds[1], delimiter=",")
