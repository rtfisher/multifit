from numpy import sqrt, pi, exp, linspace, loadtxt, log
import scipy
import numpy as np
from lmfit import  Model

import matplotlib.pyplot as plt
import sys 

# Define functions to be fit. The first argument is always the dependent variable(s),
#  followed by the function parameters to be fit.

from functions_lmfit import linear, powerlaw, exponential, gaussian, lognormal, poisson, logpoisson

# Attempt a fit to a set of standard trial functions, and determine a best
#  fit according to a reduced chi-squared criterion.

def multifit (x, y, yerr):

  funclist = [linear, powerlaw, exponential, gaussian, lognormal, poisson, logpoisson] 

# Take best reduced chi value to be Inf initially.

  bestredchi = float ('inf')

  for func in funclist :
    print ("Fitting to function...", func)
    gmodel = Model (func)
    result = gmodel.fit (y, x = x, weights = yerr)
    print ('Reduced chi-squared = ', result.redchi)
    if (result.redchi < bestredchi):  # if redchi better than best redchi yet
      bestresult = result
      bestredchi = result.redchi

  print (bestresult.fit_report() )

# Plot model data, alongside the initial fit determined by default parameters,
#  as well as the best fit.

  bestresult.plot (yerr = yerr)
#  plt.plot(x, y,         'bo')
  plt.legend( ['Data'])
  plt.plot(x, bestresult.init_fit, 'k--')
  plt.legend( ['Initial Fit'] )
#  plt.plot(x, bestresult.best_fit, 'r-')
#  plt.legend( ['Best Fit'] )
  plt.show()
