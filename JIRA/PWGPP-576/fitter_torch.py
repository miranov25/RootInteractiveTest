# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:34:01 2020

@author: majoi
"""

import torch
import torch.nn
import torch.optim

import math
import numpy as np
import pandas as pd

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
        else: # sigma.shape == y.shape:
            lossfunc = mse
        #else:
        #    lossfunc = mse_matrix_sigma
    if sigma is None:
        invsigma = 1
    else:
        invsigma = 1/sigma

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
        
        stall = torch.max(torch.abs(newparams-oldparams)) < xtol
        
        oldparams = newparams
        
        if optcond or stall or math.isnan(torch.sum(newparams).item()):
            break
   # print(i)        
    optimizer.zero_grad()
    with torch.no_grad():
        y_fit = fitfunc(x,*params)
        loss = lossfunc(y,y_fit,weights,invsigma)
    
    if len(params) == 1:
        hessian = torch.autograd.functional.hessian(lambda a:lossfunc(y,fitfunc(x,a),weights,invsigma),params[0])
        return params,hessian.detach().pinverse()*loss
    else:
        hessian = torch.autograd.functional.hessian(lambda *a:lossfunc(y,fitfunc(x,*a),weights,invsigma),tuple(params))
        return params,hessian.detach().pinverse()*loss
    
    return params,hessian.detach().pinverse()*loss

def curve_fit_BS(x,y,fitfunc,init_params,sigma0=1,weights=None,nbootstrap=50,fitter_options={},fitter_name='Pytorch_LBFGS'):

    weights_idx=[]
    fitted_params = []
    errors=[]

    n = y.shape[0]
    
    if weights is None:
        weights = bootstrap_weights(nbootstrap,n)
        
    if not torch.is_tensor(weights):
        weights = torch.from_numpy(weights)

    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    
    for i in range(nbootstrap):
        p0 = init_params[i]
        if torch.is_tensor(p0):
            p_init = [p0.clone().detach().requires_grad_(True)]
        else:
            p_init = [torch.tensor(i,requires_grad=True) for i in p0]
        p,q = curve_fit(fitfunc,x,y,p_init,weights=weights[i],sigma=sigma0,**fitter_options)
        fitted_params.append(np.concatenate([j.detach() for j in p]))
        errors.append(np.sqrt(np.diag(q)))
        weights_idx.append(i)
        
    df = create_benchmark_df(fitter_name,fitted_params,errors,n,weights_idx)    
    return df,weights

def create_benchmark_df(optimizers,params,covs,npoints,idx):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'optimizers':optimizers,'number_points':npoints,'weights_idx':idx}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_weights(nfits,npoints):
    return torch.stack([torch.bincount(torch.randint(0,npoints,[npoints]),minlength=npoints) for i in range(nfits)])

