import numpy as np
import pandas as pd

def get_stock_symbols(stock_data):
    '''takes in stock data, returns numpy array of stock names'''
    stock_names = []
    for i in stock_data.values:
        if(i[0] not in stock_names):
            stock_names.append(i[0])
    return stock_names

def particular_stock_data(stock_data, stock_names_array, index):
    '''in stock data, stock name array and index of the stock
        returns a pandas DataFrams of the specified stock's entire
        monthly data'''
    start_index = 21 * index
    particular_stock_data = pd.DataFrame(stock_data, 
                            index=[stock_data.axes[0][start_index:start_index+21]],
                            columns=stock_data.axes[1])
    return particular_stock_data


# needs to be a 2D dataframe. It is treated as
# if it is one stock's data for the month. Passing anything else
# in will likely output nonsense. It returns a 1D numppy array of all of
# the stock's monthly data.
def stock_data_in_one_line(stock_data, stock_names, index):
    ''' takes in stock data, stock names array and index of the stock
        returns the stock's monthly data in a 1D numpy array'''
    particular_stock = particular_stock_data(stock_data, stock_names, index)
    single_line_data = []
    for i in range(0, particular_stock.axes[0].size):
        single_line_data.append(particular_stock.values[i][3:])
    return np.ravel(single_line_data)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)


# computes forward message using standard algorithm
#TODO: make sure indices are correct
def compute_forward_messages(data_size, pi_0, A, L):

	# instantiate messages
	forward_messages = np.zeros((5, data_size))

	# initialize first messages
	for i in range(5):
		forward_messages[0][i] = pi_0[i]
		
	# recursive step
	for t in range(1, 5):
		for i in range(data_size):
			for j in range(5):
				forward_messages[t][i] += forward_messages[t-1][i] * A[j][i]

			forward_messages[t][i] *= L[i][t]
	
	return forward_messages

	
# computes backward message using standard algorithm
#TODO: make sure indices are correct
def compute_backward_messages(data_size, A, L):

	# instantiate messages
	backward_messages = np.zeros((5, data_size))
		
	# initialize given values
	for i in range(5):
		backward_messages[i][i] = 0

	# recursive step
	for t in range(1, 5):
		for i in range(data_size):
			for j in range(5):
				backward_messages[5-t][i] += backward_messages[6-t][i] * \
														A[j][t] * L[t][j]
	
	return backward_messages
