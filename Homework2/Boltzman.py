import numpy as np
import matplotlib.pyplot as plt

class Boltzman:
	P = lambda x: 1 / (1+np.exp(-2 * x))

	def __init__(self, n_visible, n_hidden):
		self.visible_neurons = np.zeros((n_visible, 1))
		self.hidden_neurons = np.zeros((n_hidden, 1))
		self.init_paramters()

	def init_paramters(self):
		self.weights = np.random.rand(
			self.hidden_neurons.shape[0],
			self.visible_neurons.shape[0]
			)
		self.tresholds_hidden = np.random.rand(self.hidden_neurons.shape[0], 1)
		self.tresholds_visible = np.random.rand(self.visible_neurons.shape[0], 1)

	def update(self, input_data, n_iter, n_time, learning_rate=None, training_mode=False):
		"""
		input_data: each column is a pattern
		"""
		if not training_mode and input_data.shape[1] != 1:
			error("More than one pattern in input data while not in training mode")
		for i in range(n_iter):
			# Go through input data
			for j in range(input_data.shape[1]):
				self.visible_neurons[:] = input_data[:,j] # Init visible neurons
				initial_visible_neurons = self.visible_neurons.copy()

				# Init hidden neurons
				initial_hidden_local_fields = self.weights @ self.visible_neurons - self.tresholds_hidden
				hidden_local_fields = initial_hidden_local_fields
				for k in range(self.hidden_neurons.shape[0]):
					r = np.random.rand()
					if r < Boltzman.P(hidden_local_fields[k]):
						self.hidden_neurons[k] = 1
					else:
						self.hidden_neurons[k] = -1
				# Update neuron states
				for t in range(n_time):
					# Update visible neurons
					visible_local_fields = (self.hidden_neurons.T @ self.weights).T - self.tresholds_visible

					for k in range(self.visible_neurons.shape[0]):
						r = np.random.rand()
						if r < Boltzman.P(visible_local_fields[k]):
							self.visible_neurons[k] = 1
						else:
							self.visible_neurons[k] = -1

					# Update hidden neurons
					hidden_local_fields = self.weights @ self.visible_neurons - self.tresholds_hidden
					for k in range(self.hidden_neurons.shape[0]):
						r = np.random.rand()
						if r < Boltzman.P(hidden_local_fields[k]):
							self.hidden_neurons[k] = 1
						else:
							self.hidden_neurons[k] = -1
				final_hidden_local_fields = hidden_local_fields
				final_visible_neurons = self.visible_neurons.copy()

				if training_mode:
					# Increments weights and tresholds
					dw =  np.outer(np.tanh(initial_hidden_local_fields), initial_visible_neurons)
					dw -= np.outer(np.tanh(final_hidden_local_fields), final_visible_neurons)
					dt_visible = initial_visible_neurons - final_visible_neurons
					dt_hidden = np.tanh(initial_hidden_local_fields) - np.tanh(final_hidden_local_fields)
					self.weights += learning_rate * dw
					self.tresholds_visible -= learning_rate * dt_visible
					self.tresholds_hidden  -= learning_rate * dt_hidden
		if not training_mode:
			return np.sign(self.visible_neurons)

N_HIDDEN = 8
N_VISIBLE = 3
patterns = np.matrix(
	[[-1, -1, -1],
	[1, 1, -1],
	[-1, 1, 1],
	[1, -1, 1]]
	).T

n_iter = 100
n_time = 100
learning_rate = 0.1
network = Boltzman(N_VISIBLE, N_HIDDEN)
network.update(patterns, n_iter, n_time, learning_rate=learning_rate, training_mode=True)
for i in range(patterns.shape[1]):
	pred = network.update(patterns[:,i], 1, n_time)
	print("Input: {}\tPrediction: {}".format(patterns[:,i].T, pred.T))