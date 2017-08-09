import pandas as pd #data manipulation
import keras as ks #machine learning
from keras.layers import Dense
import numpy as np #numerical
# %matplotlib inline
from matplotlib import pyplot as plt #graphing
import math
import matplotlib as mpl
print(mpl.__version__)

stock_data = pd.read_csv("../data/mquote201010.csv")

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


def learn_stock(stock_data, stock_index):
    
    #takes in stock data and returns particular stock in one line based on name
    names_array = get_stock_symbols(stock_data)
    data = stock_data_in_one_line(stock_data, names_array, stock_index)
    
    model = ks.models.Sequential()

    # fix random seed for reproducibility
    np.random.seed(15)

    # split data into test and training sets
    train_size = int(data.size*0.55555)
    train = data[0:train_size]
    test = data[train_size:]
    
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    # create and fit Multilayer Perceptron model
    model.add(Dense(15, input_dim=look_back, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=10, batch_size=20, verbose=2)
    
    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    testScore = model.evaluate(testX, testY, verbose=0)
    
    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # shift train predictions for plotting
    trainPredictPlot = np.empty(data.size)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.ravel()

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1] = testPredict.ravel()
    
    # calculate difference between real and predicted stock data
    differences = [0] * data.size
    for i in range(0, trainPredictPlot.size):
        differences[i] = data[i] - trainPredictPlot[i]
    for i in range(train_size, data.size):
        differences[i] = data[i] - testPredictPlot[i]
        
    # percent error of each data point
    percent_error = [0]*data.size
    for i in range(0, trainPredictPlot.size-1):
        percent_error[i] = 100*math.fabs((data[i] - trainPredictPlot[i])/data[i])
    for i in range(train_size, data.size-1):
        percent_error[i] = 100*math.fabs((data[i] - testPredictPlot[i])/data[i])
        
    # difference of data shifted by 1 and trained data
    shifted_difference = [0]*data.size
    for i in range(0, trainPredictPlot.size-1):
        shifted_difference[i] = data[i+1] - trainPredictPlot[i]
    for i in range(train_size-1, data.size-1):
        shifted_difference[i] = data[i+1] - testPredictPlot[i]
    
    # plot baseline and predictions
    plt.style.use('ggplot') 
    fig = plt.figure(figsize=(8,17))
    fig.suptitle("Data")
    ax = plt.subplot("511")
    ax.set_title('Original Data')
    ax.plot(data, 'b')
    ax = plt.subplot("512")
    ax.set_title('Trained Model Output')
    ax.plot(trainPredictPlot, 'g')
    ax.plot(testPredictPlot, 'r')
    ax = plt.subplot("513")
    ax.set_title('Difference (Original vs. Trained Data)')
    ax.plot(differences, 'k')
    ax = plt.subplot("514")
    ax.set_title('Percent Error')
    ax.plot(percent_error, 'm')
    ax = plt.subplot("515")
    ax.set_title('Difference (Original-1 vs. Trained Data)')
    plt.plot(shifted_difference, 'c')
    plt.show()
    print(trainScore)
    print(testScore)
    
learn_stock(stock_data, 400)
