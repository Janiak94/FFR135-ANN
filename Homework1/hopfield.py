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

    def update(self, n_iter, update_scheme="sync", initial_state=None):
        if initial_state is not None: # If we want to feed in another pattern
            self.set_state(initial_state)
        has_changed = False
        if update_scheme == "async" or update_scheme == "async_typewriter":
            # Perform async updates
            for i in range(n_iter):
                if update_scheme == "async":
                    neuron = np.random.randint(self.pattern_size) # Choose random neuron to update
                else:
                    neuron = i % self.pattern_size # Use typewriter scheme (pattern is flattened so this works)
                old_val = self.state.pattern[neuron]
                self.state.pattern[neuron] = Hopfield._sgn(self.weights[neuron,:] @ self.state.pattern) # Update single neuron state
                if has_changed is False and self.state.pattern[neuron] != old_val: # Check if neuron was flipped
                    has_changed = True
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


def one_step_error(zero_diag):
    # First task
    N_TRIALS = 100
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Task number", dest="task_number")

    first_parser = subparsers.add_parser("1", help="The first task")
    first_parser.add_argument("-z", help="Flag if diagonals in the weight matrix in the first task should be zero", action="store_true", dest="zero_diag")

    second_parser = subparsers.add_parser("2", help="The second task")
    second_parser.add_argument("pattern_number", help="Which pattern to use", choices=["1","2","3"])

    # Add other parsers
    args = parser.parse_args()
    if args.task_number == "1":
        one_step_error(args.zero_diag)
    if args.task_number == "2":
        recognising_digits(args.pattern_number)




