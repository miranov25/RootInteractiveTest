# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:34:01 2020

@author: majoi
"""

import torch
import torch.nn
import torch.optim

def mse(x,y):
    """
    standard loss: mean squared error

    @param y_pred: predicted value
    @param y_true: target value
    @return: mse
    """
    return torch.mean((x-y)**2)



def curve_fit(fitfunc,x,y,params, weights = None, lossfunc = None, optimizer = torch.optim.LBFGS, ytol = 1e-5, xtol = 1e-5, max_steps = 1, optimizer_options={}):
    """
    curve fitting

    @param fitfunc: the function to be fitted
    @param x: input parameters
    @param y: output parameters
    @param params: the parameters that should be fitted, filled with the initial guess values
    @return: values of fitted parameters
    """
    
    optimizer = optimizer(params, **optimizer_options)
        
    if lossfunc is None:
        if weights is None:
            lossfunc = mse
        elif weights.shape == y.shape:
            lossfunc = lambda x,y:torch.sum(((x-y)/weights)**2)
        else:
            t = weights.cholesky()
            lossfunc = lambda x,y:torch.sum(torch.cholesky_solve((x-y).reshape([-1,1]),t)**2)
            
    oldparams = torch.cat([i.flatten() for i in params])

    #for i in range(options["max_iterations"]):
    for i in range(max_steps):
        
        def closure():
            optimizer.zero_grad()
            y_appx = fitfunc(x,*params)
            loss = lossfunc(y,y_appx)
            loss.backward()
            return loss
        
        
        
        optimizer.step(closure)
    
        optcond = max(torch.max(torch.abs(i.grad)) for i in params) < ytol
        
        newparams = torch.cat([i.flatten() for i in params])
        
        stall = max(torch.max(torch.abs(newparams-oldparams)) for i in params) < xtol
        
        oldparams = newparams.clone()
        
        if optcond or stall:
            break
   # print(i)        
    with torch.no_grad():
        y_fit = fitfunc(x,*params)
        loss = lossfunc(y,y_fit)
    
    
    return params,loss,i