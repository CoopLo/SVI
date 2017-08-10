## SVI
Implementation of the algorithm outlined here: http://proceedings.mlr.press/v32/johnson14.pdf. Specifically using stochastic variational inference to fit a hidden markov model to minute level stock data.

I am currently trying the code found here: https://github.com/dillonalaird/pysvihmm. In pysvihmm.hmmbase, I commented out line 16: import cPickle as pkl because it is not used anywhere and I didn't want to bother with installing cPickle. I also found that compiling the cython files did not work using the command given in the README so I added "import pyximport; pyximport.install(), modified line 26 to be: import pysvihmm.hmm_fast as hmm_fast and modified line 27 to be: import pysvihmm.util as util. Turns out I needed to run python setup.py build_ext --inplace in pysvihmm as well. Line 3 of util.py changed from munkres... to from pysvihmm.munkres... In line 412 of hmmbse.py, got rid of middle argument, None because it doesn't do anything. If I remember correctly there is another bug I had to fix regarding passing parameters around incorrectly. In one function call the parameters are in one order and in the function definition they are in a different order. The error it threw was calling numpy array properties on a python list. I don't remember where exactly this occurred.

Some of the formatting is off because I use 3 space tabs in vim and github doesn't like that.

my_implementation contains my implementations of a hidden markov model class and the algorithm in the paper linked above.
training_particular_stock.py is a keras model being used to model stock prices, link to the tutorial I followed is at the top of the file. The other files are either written by me or copied from pysvihmm. I will organize this all later.

Dependencies: cython, pybasicbayes, numpy, pandas

Currently: I got the error that numpy couldn't do the cholesky decomposition on the variance of the gaussian emmissions because it was not a matrix (and definitely not a hermitian positive definite matrix), just a float. I got around this by throwing the variance in some brackets to make it a 1x1 2D matrix. The output is garbage so far because no inference step has been performed.

# SVI Model Output vs. Actual Stock Data
![svi_output](https://user-images.githubusercontent.com/17442830/29154144-f8bd8bfa-7d57-11e7-8b29-ffb08ed9f72a.png)

# Keras Model Output vs. Actual Stock Data
![keras_output](https://user-images.githubusercontent.com/17442830/29154335-67e9913a-7d59-11e7-9032-5b65b556e62f.png)
