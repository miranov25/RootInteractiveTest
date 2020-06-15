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
    return torch.sum(weights*((invsigma*(x-y))**2))

def mse_matrix_sigma(x,y,weights,invsigma):
    eps = x-y
    v = torch.cholesky_solve(eps,invsigma)
    return (weights*eps).dot(v)

def curve_fit(fitfunc,x,y,params, weights = 1, sigma = None, lossfunc = None, absolute_sigma=False, optimizer = 'lbfgs', ytol = 1e-5, xtol = 1e-5, max_steps = 20, optimizer_options={},verbose=False):
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

 #   if weights is None:
 #       weights = 1/y.size(0)

    oldparams = torch.cat([i.flatten() for i in params])
    nparams = oldparams.shape[0]

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
    else:
        hessian = torch.autograd.functional.hessian(lambda *a:lossfunc(y,fitfunc(x,*a),weights,invsigma),tuple(params))
        hessian = torch.stack([torch.stack([hessian[j][i] for i in range(len(params))]) for j in range(len(params))],1)

    pcov = 2*hessian.detach().pinverse()
    if not absolute_sigma:
        if y.size(0) > nparams:
            chisq = loss / (y.size(0) - nparams)
            pcov = pcov * chisq
        else:
            pcov.fill(np.inf)
            
    if verbose:
        return params,pcov,optimizer.state[params[0]]["func_evals"]
    return params,pcov

def curve_fit_BS(x,y,fitfunc,init_params,sigma0=1,weights=None,nbootstrap=50,fitter_options={},device=None,fitter_name='Pytorch_LBFGS'):

    weights_idx=[]
    fitted_params = []
    errors=[]
    chisq=[]
    chisq_transformed=[]
    niter=[]

    npoints = y.shape[0]

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
        p,q,n = curve_fit(fitfunc,x,y,p_init,weights=weights[i],sigma=sigma0,**fitter_options,verbose=True)
        fitted_params.append(np.hstack([j.detach().cpu().numpy() for j in p]))
 #       fitted_params.append(torch.cat([j.detach() for j in p]).cpu().numpy())
        errors.append(np.sqrt(np.diag(q.cpu().numpy())))
        niter.append(n)
        weights_idx.append(i)
        with torch.no_grad():
            y_fit = fitfunc(x,*p)
            loss = torch.sum(((y-y_fit)/sigma0)**2)
            loss_transformed = torch.sum(weights[i]*((y-y_fit)/sigma0)**2)
        chisq.append(loss.cpu().numpy())
        chisq_transformed.append(loss_transformed.cpu().numpy())
    params = np.stack(fitted_params)
    mean = np.mean(params,0)
    median = np.median(params,0)
    std = np.std(params,0)

    df = create_benchmark_df(fitter_name,fitted_params,errors,npoints,weights_idx,chisq,chisq_transformed,niter)
    return df,mean,median,std,weights

def create_benchmark_df(optimizers,params,covs,npoints,idx,chisq,chisq_transformed,niter):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'optimizers':optimizers,'number_points':npoints,'weights_idx':idx,'chisq':chisq,'chisq_transformed':chisq_transformed,'n_iter':niter}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_weights(nfits,npoints):
    return torch.stack([torch.bincount(torch.randint(0,npoints,[npoints]),minlength=npoints) for i in range(nfits)])
