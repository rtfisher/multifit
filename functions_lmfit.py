from numpy import sqrt, pi, exp, linspace, log
import scipy

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
#      print ("Poisson Error: lambda < 0.")
#      sys.exit()
       lambd = 0.
    return (amp * exp (-lambd) * lambd**x / (scipy.special.gamma (x)) )

def logpoisson(x, lambd = 1., amp = 1.):
    "continuous log Poisson: logpoisson (x, lambd, amp)"
    lnx = log (x)
    if (lambd < 0):
#      print ("Poisson Error: lambda < 0.")
#      sys.exit()
       lambd = 0.
    return  (amp * exp (-lambd) * lambd**lnx / (x * scipy.special.gamma (lnx)) )

