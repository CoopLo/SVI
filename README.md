## SVI
Implementation of the algorithm outlined here: http://proceedings.mlr.press/v32/johnson14.pdf. Specifically using stochastic variational inference to fit a hidden markov model to minute level stock data.

I am currently trying the code found here: https://github.com/dillonalaird/pysvihmm 

Dependencies: cython, pybasicbayes, numpy, pandas

Currently: I got the error that numpy couldn't do the cholesky decomposition on the variance of the gaussian emmissions because it was not a matrix (and definitely not a hermitian positive definite matrix), just a float. I got around this by throwing the variance in some brackets to make it a 1x1 2D matrix. The output is pretty much garbage after doing that.

# SVI Model Output vs. Actual Stock Data
![svi_output](https://user-images.githubusercontent.com/17442830/29154144-f8bd8bfa-7d57-11e7-8b29-ffb08ed9f72a.png)

# Keras Model Output vs. Actual Stock Data
![keras_output](https://user-images.githubusercontent.com/17442830/29149730-ab502600-7d3b-11e7-9096-41b3cb339b55.png)

