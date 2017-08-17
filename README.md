## SVI
Implementation of the algorithm outlined here: http://proceedings.mlr.press/v32/johnson14.pdf. Specifically using stochastic variational inference to fit a hidden markov model to minute level stock data. If you stumbled upon this accidentally I apoligize for all of the terrible coding style and whatnot. I haven't yet gone through and polished it all up. Also the data I have been using isn't public so it will not be uploaded.

I heavily modifid hmmbase.py, hmmsvi.py and util.py from pysvihmm as well as slight modifications to gaussian.py from pybasicbayes.distributions. I have not documented all of the changes here but I have uploaded all three files. Below is some of the modifications I made.
I am currently trying the code found here: https://github.com/dillonalaird/pysvihmm. In pysvihmm.hmmbase, I commented out line 16: import cPickle as pkl because it is not used anywhere and I didn't want to bother with installing cPickle. I also found that compiling the cython files did not work using the command given in the README so I added "import pyximport; pyximport.install(), modified line 26 to be: import pysvihmm.hmm_fast as hmm_fast and modified line 27 to be: import pysvihmm.util as util. Turns out I needed to run python setup.py build_ext --inplace in pysvihmm as well. Line 3 of util.py changed from munkres... to from pysvihmm.munkres... In line 412 of hmmbse.py, got rid of middle argument, None because it doesn't do anything. If I remember correctly there is another bug I had to fix regarding passing parameters around incorrectly. In one function call the parameters are in one order and in the function definition they are in a different order. The error it threw was calling numpy array properties on a python list. I don't remember where exactly this occurred. There are so many mistakes in the package that it would be annoyingly long to write them all down here.

my_implementation contains my implementations of a hidden markov model class and the algorithm in the paper linked above.
training_particular_stock.py is a keras model being used to model stock prices, link to the tutorial I followed is at the top of the file. The other files are either written by me or copied from pysvihmm. I will organize this all later.

Dependencies: cython, pybasicbayes, numpy, pandas

My current issue is that var_tran will not update in the global inference step, no matter what I do to it. This caused the output of the fitted model to be garbage. Lines 173-178 in hmmsvi.py, specifically. I left some of the debugging stuff in various files because it is still useful for me.

# SVI Model Output vs. Actual Stock Data
![more_svi_output](https://user-images.githubusercontent.com/17442830/29325099-2a908f58-81ac-11e7-804c-7ea291977f13.png)

# Keras Model Output vs. Actual Stock Data
![keras_output](https://user-images.githubusercontent.com/17442830/29154335-67e9913a-7d59-11e7-9032-5b65b556e62f.png)
