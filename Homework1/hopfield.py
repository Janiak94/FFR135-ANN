import argparse
import numpy as np

def generate_patterns(size, n):
    # Generate n patterns with size bits
    # The patterns are stored in the columns
    patterns = np.zeros((size, n))
    for i in range(n):
        patterns[:,i] = np.random.choice([-1, 1], size)
    return patterns

class Hopfield:
    _sgn = lambda x: np.sign(x) + (x == 0)

    def __init__(self, patterns, initial_state=None, zero_diag=False):
        # Initialize the hopfield network and store the patterns
        self.pattern_size = patterns.shape[0]
        self.zero_diag = zero_diag
        self.stored_patterns = patterns
        self.state = initial_state

        self.store_patterns(patterns) # Create the weight matrix

    def store_patterns(self, patterns):
        # Use Hebb's rule and store patterns
        self.weights = np.zeros([self.pattern_size, self.pattern_size])
        for i in range(patterns.shape[1]):
            self.weights += np.outer(patterns[:,i], patterns[:,i])
        self.weights /= self.pattern_size
        if self.zero_diag:
            np.fill_diagonal(self.weights, 0)

    def update(self, n_iter, async_update):
        has_changed = False
        if async_update:
            # Perform async updates
            for i in range(n_iter):
                rand_neuron = np.random.randint(self.pattern_size) # Choose random neuron to update
                old_val = self.state[rand_neuron]
                self.state[rand_neuron] = Hopfield._sgn(self.weights[rand_neuron,:] @ self.state) # Update single neuron state
                if has_changed is False and self.state[rand_neuron] != old_val: # Check if neuron was flipped
                    has_changed = True
        return has_changed


def one_step_error(zero_diag):
    # First task
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
            rand_pattern = patterns[:,np.random.choice(patterns.shape[1])] # Get at random one of the generated patterns
            network = Hopfield(patterns, rand_pattern, zero_diag=zero_diag) # Initialize network and store patterns
            has_changed = network.update(N_UPDATES, async_update=True) # Update neurons
            if has_changed:
                n_errors += 1
        n_errors /= N_TRIALS
        one_step_error_prob.append(n_errors)
    print("One-step error probability: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*one_step_error_prob))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Task number", dest="task_number")

    first_parser = subparsers.add_parser("1", help="The first task")
    first_parser.add_argument("-z", help="Flag if diagonals in the weight matrix in the first task should be zero", action="store_true", dest="zero_diag")

    second_parser = subparsers.add_parser("2", help="The second task")

    # Add other parsers
    args = parser.parse_args()
    if args.task_number == "1":
        one_step_error(args.zero_diag)




