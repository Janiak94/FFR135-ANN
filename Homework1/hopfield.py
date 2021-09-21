import argparse
import numpy as np
import patterns as pt
import matplotlib.pyplot as plt
import matplotlib.colors
import copy

def generate_patterns(size, n):
    # Generate n patterns with size bits
    # The patterns are stored in the columns
    patterns = []
    for i in range(n):
        p = Pattern(np.random.choice([-1, 1], size)) # Create as Pattern obj
        patterns.append(p)
    return patterns

class Pattern:
    # Class to handle different pattern inputs
    def __init__(self, pattern):
        """pattern can be vector or matrix
        but will be stored as a vector, that is,
        flattened if a matrix"""
        if isinstance(pattern, list) or isinstance(pattern, tuple):
            pattern = np.array(pattern) # Convert to np.ndarray in case of list or tuple
        if not isinstance(pattern, np.ndarray) and not isinstance(pattern, np.matrix):
            raise TypeError("Unsupported type {} of pattern".format(type(pattern))) # Type check pattern
        self.shape = pattern.shape
        self.pattern = pattern.flatten() # Store as flat np.ndarray
        self.pattern_length = len(self.pattern)

    def get_image(self):
        return self.pattern.reshape(self.shape) # Return with original dimensions

    def __iter__(self):
        # Function to iterate a single pattern
        return self

class Hopfield:
    _sgn = lambda x: np.sign(x) + (x == 0)

    def __init__(self, patterns, initial_state=None, zero_diag=False):
        # Initialize the hopfield network and store the patterns
        if isinstance(patterns, list):
            self.pattern_size = patterns[0].pattern_length
        elif isinstance(patterns, Pattern):
            self.pattern_size = patterns.pattern_length
        else:
            raise TypeError("Uknown type of patterns: {}".format(patterns))
        self.order_param = None
        self.zero_diag = zero_diag
        self.stored_patterns = patterns
        self.set_state(initial_state)
        self.store_patterns(patterns) # Create the weight matrix

    def set_state(self, state):
        if state is None:
            self.state = state
        elif isinstance(state, Pattern):
            self.state = Pattern(state.get_image()) # Create copy
        else:
            self.state = Pattern(state)

    def store_patterns(self, patterns):
        # Use Hebb's rule and store patterns
        self.weights = np.zeros([self.pattern_size, self.pattern_size])
        for p in patterns:
            self.weights += np.outer(p.pattern, p.pattern)
        self.weights /= self.pattern_size
        if self.zero_diag:
            np.fill_diagonal(self.weights, 0)

    def update(self, n_iter, update_scheme="async", initial_state=None, noise=None, calc_order_param=False):
        if initial_state is not None: # If we want to feed in another pattern
            self.set_state(initial_state)
        else:
            initial_state = self.state
        order_param = 0.0
        has_changed = False
        if update_scheme == "async" or update_scheme == "async_typewriter":
            # Perform async updates
            for i in range(n_iter):
                # Choose neuron to update
                if update_scheme == "async":
                    neuron = np.random.randint(self.pattern_size) # Choose random neuron to update
                elif update_scheme == "async_typewriter":
                    neuron = i % self.pattern_size # Use typewriter scheme (pattern is flattened so this works)
                old_val = self.state.pattern[neuron]
                # Update value
                if noise is not None:
                    local_field = self.weights[neuron,:] @ self.state.pattern
                    p = 1 / (1 + np.exp(-2*local_field*noise))
                    # Stochastic update of neuron
                    if np.random.rand() <= p:
                        self.state.pattern[neuron] = 1
                    else:
                        self.state.pattern[neuron] = -1
                else:
                    self.state.pattern[neuron] = Hopfield._sgn(self.weights[neuron,:] @ self.state.pattern) # Update single neuron state
                if calc_order_param:
                    order_param += self.state.pattern @ initial_state.pattern # s_i * x_i
                # Check if neuron was flipped
                if has_changed is False and self.state.pattern[neuron] != old_val:
                    has_changed = True
        self.order_param = order_param / (n_iter * self.pattern_size)
        return has_changed

def print_pattern(pattern):
    p_image = pattern.get_image()
    print("[", end="") # Start of matrix
    for i in range(pattern.shape[0]): # Rows
        print("[", end="") # Start of row
        for j in range(pattern.shape[1]-1): # Column
            print(p_image[i,j], end=",")
        if i != pattern.shape[0]-1:
            print(p_image[i,-1], end="],\n")
        else:
            print(p_image[i,-1], end="]")
    print("]")

# First task
def one_step_error(zero_diag):
    N_TRIALS = 100000
    N = 120 # Size of patterns
    N_UPDATES = 1 # Number of updates for the network in each trial
    p_vals = [12,24,48,70,100,120] # Number of patterns in each trial
    one_step_error_prob = [] # Store the estimated errors
    for p in p_vals:
        n_errors = 0
        for i in range(N_TRIALS):
            # Perform the trials
            patterns = generate_patterns(N, p) # Generate patterns to store in network
            rand_pattern = patterns[np.random.choice(len(patterns))] # Get at random one of the generated patterns
            network = Hopfield(patterns, rand_pattern, zero_diag=zero_diag) # Initialize network and store patterns
            has_changed = network.update(N_UPDATES, update_scheme="async") # Update neurons
            if has_changed:
                n_errors += 1
        n_errors /= N_TRIALS
        one_step_error_prob.append(n_errors)
    print("One-step error probability: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*one_step_error_prob))

# Second task
def recognising_digits(pattern_number):
    digits = [Pattern(x) for x in [pt.x1, pt.x2, pt.x3, pt.x4, pt.x5]]
    N_UPDATES = 100 * digits[0].pattern_length * 1
    network = Hopfield(digits, zero_diag=True)
    p1 = Pattern(
        [[1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1]] 
        )
    p2 = Pattern(
        [[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
        [1, 1, 1, 1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, 1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, 1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, 1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, 1, 1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1]] 
        )
    p3 = Pattern(
        [[1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
        [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
         [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
         [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], 
         [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1]] 
        )
    if pattern_number == "1":
        p = p1
    elif pattern_number == "2":
        p = p2
    elif pattern_number == "3":
        p = p3
    network.update(N_UPDATES, update_scheme="async_typewriter", initial_state=p)
    print_pattern(network.state)

    bw_cmap = matplotlib.colors.ListedColormap(['white', 'black']) # -1, 1
    plt.matshow(p.get_image(), cmap=bw_cmap)
    plt.matshow(network.state.get_image(), cmap=bw_cmap)
    plt.show()

# Third task
def stochastic_hopfield_network(n_patterns):
    PATTERN_SIZE = 200
    BETA = 2
    N_UPDATES = 2*100000
    N_TRIALS = 100
    order_param_list = []
    for i in range(N_TRIALS):
        patterns = generate_patterns(PATTERN_SIZE, n_patterns) # n_patterns patterns with PATTER_SIZE bits
        network = Hopfield(patterns, zero_diag=True) # Store patterns in network
        network.update(N_UPDATES, update_scheme="async", initial_state=patterns[0], noise=BETA, calc_order_param=True) # Perform updates
        order_param_list.append(network.order_param)
    print("{:.3f}".format(np.mean(order_param_list))) # Print the mean of the order parameters attained


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help="Task number", dest="task_number")

first_parser = subparsers.add_parser("1", help="The first task")
first_parser.add_argument("-z", help="Flag if diagonals in the weight matrix in the first task should be zero", action="store_true", dest="zero_diag")

second_parser = subparsers.add_parser("2", help="The second task")
second_parser.add_argument("pattern_number", help="Which pattern to use", choices=["1","2","3"])

third_parser = subparsers.add_parser("3", help="The third task")
third_parser.add_argument("number_of_patterns", help="The number of random patterns to use", type=int)

# Add other parsers
args = parser.parse_args()
if args.task_number == "1":
    one_step_error(args.zero_diag)
if args.task_number == "2":
    recognising_digits(args.pattern_number)
if args.task_number == "3":
    stochastic_hopfield_network(args.number_of_patterns)