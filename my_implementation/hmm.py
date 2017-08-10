# This is my implementation of a default HMM with 5 hidden states
# this is specific to stock data and uses notation from
# http://proceedings.mlr.press/v32/johnson14.pdf rather than standard notation
# I am a lowly undergrad, please do not get angry at me for the notation
import numpy as np
from numpy.random import normal, dirichlet
import helper_funcs as hf

class HMM():
	
	def __init__(self, data):

		# observation parameters
		mu = np.mean(data)
		sigma = np.var(data)**0.5
		self.obs_params = np.array([normal(loc=mu, scale=sigma, size=5),
											normal(loc=mu, scale=sigma, size=5),
											normal(loc=mu, scale=sigma, size=5),
											normal(loc=mu, scale=sigma, size=5),
											normal(loc=mu, scale=sigma, size=5)])
	
		# pi[0] is initial state distribution
		self.pi = np.array([dirichlet([1, 1, 1, 1, 1]),
						dirichlet([1, 1, 1, 1, 1]),
						dirichlet([1, 1, 1, 1, 1]),
						dirichlet([1, 1, 1, 1, 1]),
						dirichlet([1, 1, 1, 1, 1]),
						dirichlet([1, 1, 1, 1, 1])])

		# pi[1:] becomes transition matrix
		self.A = self.pi[1:]

		# state sequence
		self.x = np.zeros((data.size))

		# observation sequence
		self.y_t = np.zeros((data.size))

		# likelihood potentials
		self.L = np.zeros((5, data.size))

		# standard message passing
		self.forward_messages = hf.compute_forward_messages(data.size, self.pi[0], 
																self.A, self.L)
		self.backward_messages = hf.compute_backward_messages(data.size, self.A, self.L)

	def _compute_likelihood_matrix(self):
		for t in range(data.size):
			for i in range(5):
				L[t][i] = 0
		pass


if __name__ == '__main__':
	model = HMM(np.array([0, 1, 2, 3, 4]))
