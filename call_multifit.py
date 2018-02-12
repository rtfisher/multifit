#===========================================================================
#
# Multifit : A wrapper around lmfit, calling a range of common fitting
#  functions: linear, powerlaw, exponential, Gaussian, lognormal, 
#   Poisson, and log Poisson, to determine a best fit by a reduced chi-squared
#   criterion.
#
#===========================================================================

from numpy import sqrt, pi, exp, linspace, loadtxt, log
import scipy
import numpy as np
from lmfit import  Model

import matplotlib.pyplot as plt
import sys 

from functions_lmfit import gaussian
import multifit

# The number of sample points.
numpts = 20

# Equally sample x axis with numpts, then define function over this interval,
# with a small amount of added noise.

noise_amp = 0.1  # amplitude of uniform noise 

# For this example, choose a Gaussian.

x = np.linspace (0.1, 3.0, numpts)
y = gaussian (x, amp=2.0, cen=0.5, wid=0.5)
yerr = np.random.normal(size=numpts, scale=noise_amp)
y = y + yerr

result = multifit.multifit (x, y, yerr)

