# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:34:01 2020

@author: majoi
"""

import torch
import torch.nn
import torch.optim

import math

def mse(x,y,weights,invsigma):
    """
    standard loss: mean squared error

    @param y_pred: predicted value
    @param y_true: target value
    @return: mse
    """
    return torch.sum(weights*(invsigma*(x-y)**2))

def mse_matrix_sigma(x,y,weights,invsigma):
    eps = x-y
    v = torch.cholesky_solve(eps,invsigma)
    return (weights*eps).dot(v)

def curve_fit(fitfunc,x,y,params, weights = None, sigma = None, lossfunc = None, optimizer = 'lbfgs', ytol = 1e-5, xtol = 1e-5, max_steps = 20, optimizer_options={}):
    """
    curve fitting

    @param fitfunc: the function to be fitted
    @param x: input parameters
    @param y: output parameters
    @param params: the parameters that should be fitted, filled with the initial guess values
    @return: values of fitted parameters
    """
    
    if optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS
        optimizer_options.update({'tolerance_grad':ytol,'tolerance_change':xtol,'max_iter':max_steps})
        max_steps = 1
    
    optimizer = optimizer(params, **optimizer_options)
        
    if lossfunc is None:
        if sigma is None:
            lossfunc = mse
        elif sigma.shape == y.shape:
            lossfunc = mse
        else:
            lossfunc = mse_matrix_sigma
    if sigma is None:
        invsigma = 1
    elif sigma.shape == y.shape:
        invsigma = 1/sigma
    else:
        invsigma = sigma.cholesky()

    if weights is None:
        weights = 1/y.size(0)
    
    oldparams = torch.cat([i.flatten() for i in params])

    #for i in range(options["max_iterations"]):
    for i in range(max_steps):
        
        def closure():
            optimizer.zero_grad()
            y_appx = fitfunc(x,*params)
            loss = lossfunc(y,y_appx,weights,invsigma)
            loss.backward()
            return loss
        
        
        
        optimizer.step(closure)
    
        optcond = max(torch.max(torch.abs(i.grad)) for i in params) < ytol
        
        newparams = torch.cat([i.flatten() for i in params])
        
        stall = max(torch.max(torch.abs(newparams-oldparams)) for i in params) < xtol
        
        oldparams = newparams
        
        if optcond or stall or math.isnan(params[0].data[0].item()):
            break
   # print(i)        
    optimizer.zero_grad()
    with torch.no_grad():
        y_fit = fitfunc(x,*params)
        loss = lossfunc(y,y_fit,weights,invsigma)
    
    if len(params) == 1:
        hessian = torch.autograd.functional.hessian(lambda a:lossfunc(y,fitfunc(x,a),weights,invsigma),params[0])
    else:
        hessian = torch.autograd.functional.hessian(lambda a:lossfunc(y,fitfunc(x,*a),weights,invsigma),tuple(params))
    
    return params,hessian.detach().pinverse()*loss