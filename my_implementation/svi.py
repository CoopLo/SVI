import hmm
import numpy as np
from numpy import random as rand
import pandas as pd
import helper_funcs as hf
from matplotlib import  pyplot as plt

debug = True
components = 4
mixtures = 5

# read in data
stock_data = pd.read_csv("../data/mquote201010.csv")
stock_symbols = hf.get_stock_symbols(stock_data)
single_stock = hf.stock_data_in_one_line(stock_data, stock_symbols, 400)

# break up stock data into train and test sets
train_size = int(0.5*single_stock.size)
train_set = np.asarray(single_stock[:train_size])
test_set = np.asarray(single_stock[train_size:])

# add extra dimension's data
train_index = [i for i in range(train_set.size)]
test_index = [i for i in range(test_set.size)]

# putting two dimensions together into columns
train_data = np.column_stack([train_index, train_set])
test_data = np.column_stack([test_index, test_set])

# fitting the model
model = hmm.HMM(train_data)

	
# update parameters
def _update_parameters(transmat, data, transmat_rows, debug):
	#TODO: probably ask Will at this point because I'm stuck
	pass


# initialize sufficient statistics
# n_components is the number of gaussian emmissions in the model
def _initialize_sufficient_statistics(n_components, data_size):
	
	t_y = np.zeros((n_components, data_size))
	t_trans = np.zeros((n_components, n_components))
	t_init = np.zeros((n_components))
	
	return t_y, t_trans, t_init


# accumulate sufficient statistics
def _accumulate_sufficient_statistics(model, stats):

	# compute normalizing constant
	normalizer = 0
	for i in range(model.forward_messages[0].size):
		normalizer += mode.forward_messages[forward_messages[0].size-1][i]

	# compute t_y
	for i in range(model.forward_messages[0].size-1):
		for t in range(1, model.forward_messages[0].size):
			stats[0][i] += model.forward_messages[t][i] * model.backward_messages[t][i]
		#TODO: Figure out what the fuck goes next

	# compute t_trans
	for i in range(model.update_transmat.shape[0]):
		for j in range(model.update_transmat.shape[0]):
			for t in range(1, model.forward_messages.size-1):
				stats[1][i][j] += model.forward_messages[t][i] * model.update_transmat[i][j] \
									* model.update_liklihood[t+1][j] * model.backward_messages[t+1][j] \
									/ normalizer

	# compute t_init
	for i in range(stats[2].size):
		stats[2][i] = update_pi_0 * model.backward_messages[i] / normalize

	return stats


# do svi steps
def _svi_step(original_parameters, updated_parameters, stats, step_seq):

	for i in range(step_seq.size):
		# natural parameter. Parameter of q(A)
		updated_parameters[0] = (1 - step_seq[i]) * updated_parameters[0] + \
								step_seq[i] * (originial_parameters[0] + s * stats[0][i])
		
		# alpha_i. Parameter of q(pi_0)
		updated_parameters[1] = (1 - step_seq[i]) * updated_parameters[1] * \
								step_seqq[i] * (original_parameters[1] + s * stats[1][i])

		# alpha_0. Parameter of q(theta)
		updated_parameters[2] = (1 - step_seq[i]) * updated_parameters[2] * \
								step_seqq[i] * (original_parameters[2] + s * stats[2][i])

	return updated_parameters

# takes in model and fits parameters to data using SVI
def svi_fit(model, data, iterations=100):

	stats = _initialize_sufficient_statistics(model.obs_params.shape[0], data.size)
	suf_stats = _accumulate_sufficient_statistics(model, stats)
	step_seq = [1/i for i in range(2, iterations)]

	#TODO: figure out how to do all of these
	'''
		update parameters to new HMM posterior
		finish accumulating sufficient statistics
		debug
		pray everything is how its supposed to be
	'''

	if(debug): 
		print(model.transmat_)
		print(step_size_seq)
	
if __name__ == '__main__':
	svi_fit(model, train_data, iterations=50)


