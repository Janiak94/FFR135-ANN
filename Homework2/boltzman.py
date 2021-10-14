import numpy as np
import matplotlib.pyplot as plt

class Boltzman:
	P = lambda x: 1 / (1+np.exp(-2 * x))

	def __init__(self, n_visible, n_hidden):
		self.visible_neurons = np.zeros((n_visible, 1))
		self.hidden_neurons = np.zeros((n_hidden, 1))
		self.init_paramters()

	def init_paramters(self):
		self.weights = 2*np.random.rand(
			self.hidden_neurons.shape[0],
			self.visible_neurons.shape[0]
			) - 1
		self.tresholds_hidden = 2*np.random.rand(self.hidden_neurons.shape[0], 1) - 1
		self.tresholds_visible = 2*np.random.rand(self.visible_neurons.shape[0], 1) - 1

	def predict(self, pattern, k):
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

	def train(self, input_data, n_iter, k, learning_rate):
		"""
		input_data: each column is a pattern
		"""
		p0 = 4
		input_data = input_data.copy()
		for i in range(n_iter):
			if (i % 100 == 0):
				print(r"Iteration {}/{}".format(i, n_iter))
			# Sample patterns from input_data
			patterns = input_data[:, np.random.choice(input_data.shape[1], p0)]
			for j in range(p0):
				pattern = patterns[:,j]
				self.predict(pattern, k) # Update patterns k times
					
				# Increments weights and tresholds
				dw =  np.outer(np.tanh(self.initial_hidden_local_fields), self.initial_visible_neurons)
				dw -= np.outer(np.tanh(self.final_hidden_local_fields), self.final_visible_neurons)
				dt_visible = self.initial_visible_neurons - self.final_visible_neurons
				dt_hidden = np.tanh(self.initial_hidden_local_fields) - np.tanh(self.final_hidden_local_fields)

				self.weights += learning_rate * dw
				self.tresholds_visible -= learning_rate * dt_visible
				self.tresholds_hidden  -= learning_rate * dt_hidden
		
		return self.visible_neurons

N_HIDDEN = 4
N_VISIBLE = 3
patterns = np.matrix(
	[[-1, -1, -1],
	[1, 1, -1],
	[-1, 1, 1],
	[1, -1, 1]]
	).T

n_iter = 100
k = 100
learning_rate = 0.1
network = Boltzman(N_VISIBLE, N_HIDDEN)
print(network.weights)
network.train(patterns, n_iter, k, learning_rate)
print(network.weights)
for i in range(patterns.shape[1]):
	pred = network.predict(patterns[:,i], k)
	print("Input: {}\tPrediction: {}".format(patterns[:,i].T, pred.T))