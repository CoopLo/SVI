#import hmm
import numpy as np
from numpy import random as rand
from numpy.random import normal
import pandas as pd
import helper_funcs as hf
from pysvihmm import hmmsvi, hmmsgd_metaobs
from pybasicbayes.distributions import gaussian
import matplotlib
from matplotlib import  pyplot as plt

debug = True
components = 4
mixtures = 5
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

# fitting the model
prior_init = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
# observation parameters
mu = np.asarray([np.mean(single_stock)])
sigma = np.var(single_stock)**0.5

#print(sigma.shape)
#obs_params = np.array([normal(loc=mu, scale=sigma, size=5),
                       #normal(loc=mu, scale=sigma, size=5),
                       #normal(loc=mu, scale=sigma, size=5),
                       #normal(loc=mu, scale=sigma, size=5),
                       #normal(loc=mu, scale=sigma, size=5)])
	
# pi[0] is initial state distribution

prior_tran = np.asarray(([0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]))

state=0
# get prior_tran from data
for i in range(1, train_size):
    if(train_data[i] < (1.0025*train_data[i-1]) and train_data[i] > (0.9975*train_data[i-1])):
        prior_tran[0][state] += 1
        state = 0
    elif(train_data[i] >= (1.0025*train_data[i-1]) and train_data[i] < (1.01*train_data[i-1])):
        prior_tran[1][state] += 1
        state = 1
    elif(train_data[i] >= (1.01*train_data[i-1])):
        prior_tran[2][state] += 1
        state = 2
    elif(train_data[i] <= (0.9975*train_data[i-1]) and train_data[i] > (0.99*train_data[i-1])):
        prior_tran[3][state] += 1
        state = 3
    elif(train_data[i] <= (0.99*train_data[i-1])):
        prior_tran[4][state] += 1
        state = 4


for i in range(5):
    for j in range(5):
        #print("prior_tran["+str(i)+"]["+str(j)+"]: " + str(prior_tran[i][j]))
        prior_tran[i][j] = float(prior_tran[i][j])/train_size

#print(str(prior_tran))
train_data = np.column_stack([train_index, train_set])

#sigma = [[sigma]]
#sigma = np.cov(train_data)
#print(sigma)

# PARAMETERS FOR GAUSSIAN, TAKEN FROM TEST FILE
kappa_0 = 1
nu_0 = 4
# prior emissions are gaussian
print("here?")
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

#print(prior_tran)
mu_0 = np.zeros(5)
sigma_0 = 0.75 * np.cov(obs.T)
kappa_0 = 0.01
nu_0 = 4
prior_emit = [gaussian.Gaussian(mu_0=mu_0, sigma_0=sigma_0, kappa_0=kappa_0, nu_0=nu_0)
              for _ in range(5)]
prior_emit = np.array(prior_emit)
print("here?")
model = hmmsgd_metaobs.VBHMM(obs = single_stock[:train_size],
                             prior_init = prior_init,
                             prior_tran = prior_tran,
                             prior_emit = prior_emit,
                             mb_sz = 50,
                             verbose = True)
print("MAYBE HERE")
#print(hasattr(model, "var_tran"))
# try generating before svi step
#obs_seq = model.generate_obs(single_stock[train_size:].shape[0])
#print(obs_seq[1])

# inference step needs minibatches of data. Make them here.
buffer_length = 10
minibatches = np.ndarray((int(train_size/50), 50))
#print(train_size)
for i in range(int(train_size/50)):
    for j in range(50):
        minibatches[i][j] = train_set[50*i + j]
    #print(str(minibatches[i]))

# inference step
model.infer()

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

