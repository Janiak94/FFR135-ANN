import numpy as np
import matplotlib.pyplot as plt

def simulate_boltzmann(patterns, network, n_trials, n_steps):
	"""
	Sample the state distribution from network
	"""
	pb = np.zeros((patterns.shape[1]))
	# Run trials
	for i in range(n_trials):
		r = np.random.choice(patterns.shape[1])
		pattern = patterns[:, r]
		# Step from initial pattern
		for j in range(n_steps):
			pattern = network.predict(pattern, 1)
			# Check which pattern we match
			for k in range(patterns.shape[1]):
				if np.array_equal(patterns[:,k], pattern[:]):
					pb[k] += 1
	return pb / (n_trials*n_steps)

class Boltzman:
	P = lambda x: 1 / (1+np.exp(-2 * x))

	def __init__(self, n_visible, n_hidden):
		self.visible_neurons = np.zeros((n_visible, 1), dtype=np.int)
		self.hidden_neurons = np.zeros((n_hidden, 1), dtype=np.int)
		self.init_paramters()

	def init_paramters(self):
		"""
		Init weights and tresholds
		"""
		self.weights = 2*np.random.normal(size=(self.hidden_neurons.shape[0],self.visible_neurons.shape[0]))
		self.tresholds_hidden = np.zeros((self.hidden_neurons.shape[0], 1))
		self.tresholds_visible = np.zeros((self.visible_neurons.shape[0], 1))

	def predict(self, pattern, k):
		"""
		Runs the dynamics for k steps
		"""
		self.visible_neurons[:] = pattern[:] # Init visible neurons
		# Init hidden neurons (h(t=0))
		hidden_local_fields = self.weights @ self.visible_neurons - self.tresholds_hidden
		self.initial_hidden_local_fields = hidden_local_fields.copy() # Save for training
		self.initial_visible_neurons = self.visible_neurons.copy() # Save for training
		for i in range(self.hidden_neurons.shape[0]):
			r = np.random.rand()
			if r < Boltzman.P(hidden_local_fields[i]):
				self.hidden_neurons[i] = 1
			else:
				self.hidden_neurons[i] = -1
		# Update neuron states for pattern
		for _ in range(k):
			# Update visible neurons
			visible_local_fields = (self.hidden_neurons.T @ self.weights).T - self.tresholds_visible
			for i in range(self.visible_neurons.shape[0]):
				r = np.random.rand()
				if r < Boltzman.P(visible_local_fields[i]):
					self.visible_neurons[i] = 1
				else:
					self.visible_neurons[i] = -1
			# Update hidden neurons
			hidden_local_fields = self.weights @ self.visible_neurons - self.tresholds_hidden
			for i in range(self.hidden_neurons.shape[0]):
				r = np.random.rand()
				if r < Boltzman.P(hidden_local_fields[i]):
					self.hidden_neurons[i] = 1
				else:
					self.hidden_neurons[i] = -1
		self.final_hidden_local_fields = hidden_local_fields.copy() # Save for training
		self.final_visible_neurons = self.visible_neurons.copy() # Save for training
		return self.visible_neurons.copy()

	def train(self, input_data, n_iter, k, batch_size, learning_rate):
		"""
		Training of the Boltzmann machine.
		input_data: each column is a pattern
		"""
		input_data = input_data.copy()
		for i in range(n_iter):
			dw = 0 # Init weights and treshold increments
			dt_v = 0
			dt_h = 0
			if (i % 10 == 0): # Print progress
				print("■", end="", flush=True)
			# Sample patterns from input_data
			patterns = input_data[:, np.random.choice(input_data.shape[1], batch_size, replace=True)]
			for j in range(batch_size):
				pattern = patterns[:,j]
				self.predict(pattern, k) # Update patterns k times
					
				# Increments weights and tresholds
				dw +=  np.outer(np.tanh(self.initial_hidden_local_fields), self.initial_visible_neurons)
				dw -= np.outer(np.tanh(self.final_hidden_local_fields), self.final_visible_neurons)
				dt_v -= self.initial_visible_neurons - self.final_visible_neurons
				dt_h -= np.tanh(self.initial_hidden_local_fields) - np.tanh(self.final_hidden_local_fields)
			self.weights += learning_rate * dw
			self.tresholds_visible += learning_rate * dt_v
			self.tresholds_hidden += learning_rate * dt_h
		print("\nTraining done")
		return self.visible_neurons

n_hidden_list = [1, 2, 4, 8]

dkl_list = []
for n_hidden in n_hidden_list:
	N_VISIBLE = 3
	patterns = np.matrix(
		[[-1, -1, -1],
		[1, 1, -1],
		[-1, 1, 1],
		[1, -1, 1],
		[1, 1, 1],
		[-1, -1, 1],
		[-1, 1, -1],
		[1, -1, -1]]
		).T

	n_iter = 1000 #vMax
	batch_size = 20 #p0
	k = 100
	learning_rate = 0.1 #eta
	network = Boltzman(N_VISIBLE, n_hidden)
	network.train(patterns[:,0:4], n_iter, k, batch_size, learning_rate)

	pd = np.array([1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0])
	pb = simulate_boltzmann(patterns, network, 1000, 1000)

	dkl = 0
	for i in range(4):
		if pb[i] == 0:
			error("pb[{}] == 0".format(i))
		dkl += pd[i] * np.log(pd[i] / pb[i])
	dkl_list.append(dkl)
	print("n_hidden = {}, Dkl={:.4f}".format(n_hidden, dkl))

plt.scatter(n_hidden_list, dkl_list, color='red')
plt.plot(n_hidden_list, dkl_list, color='black')
plt.grid(True)
plt.ylabel(r"$D_{kl}$")
plt.xlabel(r"$N_{hidden}$")
plt.show()