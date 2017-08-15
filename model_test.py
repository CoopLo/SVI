#import hmm
import numpy as np
from numpy import random as rand
from numpy.random import normal
import pandas as pd
import helper_funcs as hf
import hmmsvi
from pybasicbayes.distributions import gaussian
import matplotlib
from matplotlib import  pyplot as plt

debug = True
components = 4
mixtures = 5

# read in data
stock_data = pd.read_csv("../data/mquote201010.csv")
stock_symbols = hf.get_stock_symbols(stock_data)
single_stock = hf.stock_data_in_one_line(stock_data, stock_symbols, 400)

# break up stock data into train and test sets
train_size = int(0.1*single_stock.size)
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
#print(obs_seq[1])

# inference step needs minibatches of data. Make them here.
minibatches = np.ndarray((int(train_size/10), 10))
for i in range(int(train_size/10)):
	for j in range(10):
		minibatches[i][j] = train_set[10*i + j]

# inference step
model.infer(minibatches)

# generating observation sequence after svi step
post_obs_seq = model.generate_obs(single_stock[train_size:].shape[0])
print(post_obs_seq[0])
print(post_obs_seq[1])

# plotting
plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 13})
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

ax.set_xlabel("Minute")
ax.set_ylabel("Stock Price")

ax1.set_title("Actual Stock Data")
ax1.plot(single_stock[train_size:], 'b')
ax2.set_title("SVI-non-fitted HMM Model Output")
ax2.plot(obs_seq[1], 'k')
ax3.set_title("SVI-fitted HMM model Output")
ax3.plot(post_obs_seq[1], 'g')
plt.show()

