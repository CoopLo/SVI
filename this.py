import hmm
import numpy as np
from numpy import random as rand
from numpy.random import normal
import pandas as pd
import helper_funcs as hf
import hmmsvi
from pybasicbayes.distributions import gaussian
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
prior_init = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
# observation parameters
mu = np.asarray([np.mean(single_stock)])
sigma = np.var(single_stock)**0.5
#print(sigma.shape)
obs_params = np.array([normal(loc=mu, scale=sigma, size=5),
									normal(loc=mu, scale=sigma, size=5),
									normal(loc=mu, scale=sigma, size=5),
									normal(loc=mu, scale=sigma, size=5),
									normal(loc=mu, scale=sigma, size=5)])
	
# pi[0] is initial state distribution
prior_tran = np.asarray(([1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1]))

sigma = [[sigma]]

# prior emissions are gaussian
prior_emit = [gaussian.Gaussian(mu=mu, sigma=sigma),
				gaussian.Gaussian(mu=mu, sigma=sigma),
				gaussian.Gaussian(mu=mu, sigma=sigma),
				gaussian.Gaussian(mu=mu, sigma=sigma),
				gaussian.Gaussian(mu=mu, sigma=sigma)]

# do the model
model = hmmsvi.SVIHMM(prior_init = prior_init,
					prior_tran = prior_tran,
					prior_emit = prior_emit,
					obs = single_stock[:train_size])

# try generating before svi step
obs_seq = model.generate_obs(single_stock[train_size:].shape[0])

print(obs_seq[0])
print(obs_seq[1].shape)

plt.subplot(211)
plt.plot(single_stock[train_size:])
plt.subplot(212)
plt.plot(obs_seq[1])
plt.show()

