import numpy as np
from numpy import random as rand
from numpy.random import normal
import pandas as pd
import helper_funcs as hf
from pysvihmm import hmmsvi, hmmsgd_metaobs
from pybasicbayes.distributions import gaussian
import matplotlib
from matplotlib import  pyplot as plt
from pysvihmm import PositiveDefiniteException as PDE

debug = True
index = 100

# read in data
stock_data = pd.read_csv("../data/mquote201010.csv")
stock_symbols = hf.get_stock_symbols(stock_data)
single_stock = hf.stock_data_in_one_line(stock_data, stock_symbols, index)

# break up stock data into train and test sets
train_size = int(0.2*single_stock.size)
train_set = np.asarray(single_stock[:train_size])
train_data = train_set
test_set = np.asarray(single_stock[train_size:])

# add extra dimension's data
train_index = [i for i in range(train_set.size)]
test_index = [i for i in range(test_set.size)]

# putting two dimensions together into columns
test_data = np.column_stack([test_index, test_set])
train_data = np.column_stack([train_index, train_set])

# PARAMETERS FOR GAUSSIAN, TAKEN FROM TEST FILE
kappa_0 = 1
nu_0 = 4

# prior emissions are gaussian
prior_emit = [gaussian.Gaussian(mu = np.array([0,0,0,0,0]), 
                                sigma = np.eye(5),
                                mu_0 = np.zeros(5),
                                kappa_0 = kappa_0,
                                nu_0 = nu_0),
              gaussian.Gaussian(mu = np.array([1,1,1,1,1]),
                                sigma = np.eye(5),
                                mu_0 = np.zeros(5),
                                sigma_0 = np.eye(5),
                                kappa_0 = kappa_0,
                                nu_0 = nu_0),
              gaussian.Gaussian(mu = np.array([2,2,2,2,2]),
                                sigma = np.eye(5),
                                mu_0 = np.zeros(5),
                                kappa_0 = kappa_0,
                                nu_0 = nu_0),
              gaussian.Gaussian(mu = np.array([3,3,3,3,3]),
                                sigma = np.eye(5),
                                mu_0 = np.zeros(5),
                                sigma_0 = np.eye(5),
                                kappa_0 = kappa_0,
                                nu_0 = nu_0),
              gaussian.Gaussian(mu = np.array([4,4,4,4,4]),
                                sigma = np.eye(5),
                                mu_0 = np.zeros(5),
                                sigma_0 = np.eye(5),
                                kappa_0 = kappa_0,
                                nu_0 = nu_0)]

obs = np.array([prior_emit[int(np.round(4*i/train_set.size))].rvs()[0]
                for i in range(train_set.size)])

# set up parameters with intent to burn in
mu_0 = np.zeros(5)
sigma_0 = 0.75 * np.cov(obs.T)
kappa_0 = 0.01
nu_0 = 5
prior_emit = [gaussian.Gaussian(sigma = np.eye(5), mu = np.array([_,_,_,_,_]), 
                mu_0=mu_0, sigma_0=sigma_0, kappa_0=kappa_0, nu_0=nu_0)
              for _ in range(5)]
prior_emit = np.array(prior_emit)
prior_init = np.ones(5)
prior_tran = np.ones((5,5))

# instantiate model
model = hmmsgd_metaobs.VBHMM(obs = single_stock[:train_size],
                             prior_init = prior_init,
                             prior_tran = prior_tran,
                             prior_emit = prior_emit,
                             mb_sz = 50,
                             verbose = True)

print("Model has been instantiated")

# inference step is unstable. Try until it works
worked = False 
iteration = 0
while not(worked):
    try:
        iteration += 1
        print("iteration: {}".format(iteration))
        model = hmmsgd_metaobs.VBHMM(obs = single_stock[:train_size],
                             prior_init = prior_init,
                             prior_tran = prior_tran,
                             prior_emit = prior_emit,
                             mb_sz = 50,
                             verbose = True)
        model.infer()
        worked = True
    except PDE.PositiveDefiniteException:
        pass
    

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

