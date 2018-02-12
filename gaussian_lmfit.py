from numpy import sqrt, pi, exp, linspace, loadtxt, log
import scipy
import numpy as np
from lmfit import  Model

import matplotlib.pyplot as plt
import sys 

# Define functions to be fit. The first argument is always the dependent variable(s),
#  followed by the function parameters to be fit.

def linear (x, slope=1., intercept = 1.):
    "1-d linear: linear (x, slope, intercept)"
    return slope * x + intercept

def powerlaw (x, prefactor=1., exponent =1., constant = 1.):
    "1-d powerlaw: powerlaw (x, prefactor, exponent, constant)"
    return prefactor * x**exponent + constant

def exponential (x, prefactor=1., scal=1., constant = 1.):
    "1-d exponential: exponential (x, prefactor, scal, constant)"
    return prefactor * exp (scal * x) + constant 

def gaussian(x, amp = 1., cen = 1., wid = 1.):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))

def lognormal(x, amp = 1., cen = 1., wid = 1.):
    "1-d log normal: lognormal(x, amp, cen, wid"
    lnx = log (x)
    return (amp/(x * sqrt(2*pi)*wid)) * exp(-(lnx-cen)**2 /(2*wid**2))

def poisson(x, lambd = 1., amp = 1.):
    "continous Poisson: poisson (x, lambd, amp)"
    if (lambd < 0):
      print ("Poisson Error: lambda < 0.")
      sys.exit()
    return (amp * exp (-lambd) * lambd**x / (scipy.special.gamma (x)) )

def logpoisson(x, lambd = 1., amp = 1.):
    "continuous log Poisson: logpoisson (x, lambd, amp)"
    lnx = log (x)
    return  (amp * exp (-lambd) * lambd**lnx / (x * scipy.special.gamma (lnx)) )

# The number of sample points.
numpts = 20

# Equally sample x axis with numpts, then define function over this interval,
# with a small amount of added noise.

noise_amp = 0.1  # amplitude of uniform noise 

x = np.linspace (0.1, 3.0, numpts)
y = gaussian (x, amp=2.0, cen=1.0, wid=1.5) + noise_amp * np.random.rand (numpts)
#y = lognormal (x, 1.0, 2.0, 1.0) + 0.01 * np.random.rand (numpts)
#y = poisson (x, 2.0, 1.0) + 0.1 * np.random.rand (numpts)

funclist = [linear, powerlaw, exponential, gaussian, lognormal, poisson, logpoisson] 

# Take best reduced chi value to be Inf initially.

bestredchi = float ('inf')

for func in funclist :
  print ("Fitting to function...", func)
  gmodel = Model (func)
  result = gmodel.fit (y, x = x)
  print ('Reduced chi-squared = ', result.redchi)
  if (result.redchi < bestredchi):  # if redchi better than best redchi yet
    bestresult = result
    bestredchi = result.redchi

print (bestresult.fit_report() )

# Plot model data, alongside the initial fit determined by default parameters,
#  as well as the best fit.

plt.plot(x, y,         'bo')
plt.plot(x, bestresult.init_fit, 'k--')
plt.plot(x, bestresult.best_fit, 'r-')
plt.show()
