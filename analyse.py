# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:17:03 2016

@author: lbignell
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def fixplot(title, ylab='Volume Fraction', xlab='Size (nm)', axfontsize=16,
            titsize=18, legncol=1, legfontsize=12, legloc='upper right'):
    '''
    Fixes current figure.
    '''
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=axfontsize)
    ax.yaxis.set_tick_params(labelsize=axfontsize)
    plt.legend(loc=legloc, ncol=legncol, fontsize=legfontsize)
    plt.xlabel(xlab, fontsize=axfontsize)
    plt.title(title, fontsize=titsize)
    plt.ylabel(ylab, fontsize=axfontsize)
    plt.xscale('log')
    return
    
def get_mean_std(sizes, vals):
    '''
    Returns the mean and standard deviation size of vals.
    '''
    nums = list(vals)
    nums = nums/sum(nums)
    meanlist = nums*list(sizes)
    mean = sum(meanlist)
    meanlistsq = meanlist*meanlist
    meanofsq = sum(meanlistsq)
    std = (meanofsq - mean**2)**0.5
    return mean, std

def getmedian(sizes, vals):
    '''
    returns the median size of vals.
    '''
    tot = sum(list(vals))
    running = 0
    i = 0
    while running<tot/2:
        running+=vals[i]
        i+=1
    return (sizes[i-1]+sizes[i])/2

def DoubleGauss(x, p):
    #A1, mu1, sigma1, A2, mu2, sigma2 = p
    return p[0]*np.exp(-(x-p[2])**2/(2.*p[3]**2)) + p[1]*np.exp(-(x-p[4])**2/(2.*p[5]**2))

def DoubleGauss_cf(x, p0, p1, p2, p3, p4, p5):
    #A1, mu1, sigma1, A2, mu2, sigma2 = p
    return p0*np.exp(-(x-p2)**2/(2.*p3**2)) + p1*np.exp(-(x-p4)**2/(2.*p5**2))

def fit_function(p0, datax, datay, function, sigma=None,
                 cf_func=DoubleGauss_cf, **kwargs):

    errfunc = lambda p, x, y: function(x,p) - y

    ##################################################
    ## 1. COMPUTE THE FIT AND FIT ERRORS USING leastsq
    ##################################################
    # If using optimize.leastsq, the covariance returned is the 
    # reduced covariance or fractional covariance, as explained
    # here :
    # http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
    # One can multiply it by the reduced chi squared, s_sq, as 
    # it is done in the more recenly implemented scipy.curve_fit
    # The errors in the parameters are then the square root of the 
    # diagonal elements.   
    pfit, pcov, infodict, errmsg, success = \
        leastsq( errfunc, p0, args=(datax, datay), \
                              full_output=1)

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf
    error = [] 
    for i in range(len(pfit)):
        try:
            error.append( np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    #print "cov matrix = ", pcov
    ###################################################
    ## 2. COMPUTE THE FIT AND FIT ERRORS USING curvefit
    ###################################################
    # When you have an error associated with each dataY point you can use 
    # scipy.curve_fit to give relative weights in the least-squares problem. 
    datayerrors = sigma#kwargs.get('datayerrors', None)
    curve_fit_function = cf_func#kwargs.get('curve_fit_function', function)
    if datayerrors is None:
        try:
            pfit, pcov = \
                curve_fit(curve_fit_function,datax,datay,p0=p0)
        except RuntimeError:
            pass#pfit and pcov will just be the leastsq values
    else:
        try:
            pfit, pcov = \
                curve_fit(curve_fit_function,datax,datay,p0=p0,
                          sigma=datayerrors)
        except RuntimeError:
            pass#pfit and pcov will just be the leastsq values
    error = [] 
    for i in range(len(pfit)):
        try:
            error.append( np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = np.array(error)  

    ####################################################
    ## 3. COMPUTE THE FIT AND FIT ERRORS USING bootstrap
    ####################################################        
    # An issue arises with scipy.curve_fit when errors in the y data points
    # are given.  Only the relative errors are used as weights, so the fit
    # parameter errors, determined from the covariance do not depended on the
    # magnitude of the errors in the individual data points.  This is clearly wrong. 
    # 
    # To circumvent this problem I have implemented a simple bootstraping 
    # routine that uses some Monte-Carlo to determine the errors in the fit
    # parameters.  This routines generates random datay points starting from
    # the given datay plus a random variation. 
    #
    # The random variation is determined from average standard deviation of y
    # points in the case where no errors in the y data points are avaiable.
    #
    # If errors in the y data points are available, then the random variation 
    # in each point is determined from its given error. 
    # 
    # A large number of random data sets are produced, each one of the is fitted
    # an in the end the variance of the large number of fit results is used as 
    # the error for the fit parameters. 
    # Estimate the confidence interval of the fitted parameter using
    # the bootstrap Monte-Carlo method
    # http://phe.rockefeller.edu/LogletLab/whitepaper/node17.html
    residuals = errfunc( pfit, datax, datay)
    s_res = np.std(residuals)
    ps = []
    # 100 random data sets are generated and fitted
    for i in range(100):
        if datayerrors is None:
            randomDelta = np.random.normal(0., s_res, len(datay))
            randomdataY = datay + randomDelta
        else:
            randomDelta =  np.array( [ \
                                np.random.normal(0., derr + 1e-10,1)[0] \
                                for derr in datayerrors ] ) 
            randomdataY = datay + randomDelta
        randomfit, randomcov = \
            leastsq( errfunc, p0, args=(datax, randomdataY),\
                    full_output=0)
        ps.append( randomfit ) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 
    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit


    # Print results 
    print("\nlestsq method :")
    print("pfit = ", pfit_leastsq)
    print("perr = ", perr_leastsq)
    print("\ncurvefit method :")
    print("pfit = ", pfit_curvefit)
    print("perr = ", perr_curvefit)
    print("\nbootstrap method :")
    print("pfit = ", pfit_bootstrap)
    print("perr = ", perr_bootstrap)
    return pfit_curvefit, perr_curvefit#pfit_leastsq, perr_leastsq#_bootstrap
