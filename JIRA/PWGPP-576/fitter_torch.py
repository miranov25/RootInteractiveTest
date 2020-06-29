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
    return torch.sum(weights*torch.square(invsigma*(x-y)))

def mse_matrix_sigma(x,y,weights,invsigma):
    eps = x-y
    v = torch.cholesky_solve(eps,invsigma)
    return (weights*eps).dot(v)

def curve_fit(fitfunc,x,y,params, weights = 1, sigma = None, lossfunc = None, absolute_sigma=False, optimizer = 'lbfgs', ytol = 1e-5, xtol = 1e-5, max_steps = 20, optimizer_options={},output_sigma=True,full_output=False):
    """
    curve fitting

    @param fitfunc: the function to be fitted
    @param x: input parameters
    @param y: output parameters
    @param params: the parameters that should be fitted, filled with the initial guess values
    @return: values of fitted parameters
    """

    max_iter = max_steps
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

    n_iter = 0

    fit_status = {"is_valid":True}

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
        
        if 'n_iter' in optimizer.state[params[0]]:
            n_iter += optimizer.state[params[0]]['n_iter']
        else:
            n_iter += 1

        if optcond or stall or math.isnan(torch.sum(newparams).item()) or n_iter >= max_iter:
            break
   # print(i)
   
    if n_iter >= max_iter or math.isnan(torch.sum(newparams).item()):
        if full_output:
            fit_status["is_valid"] = False
        else:
            raise RuntimeError("Optimal parameters not found: maximum number of iterations exceeded")
   
    optimizer.zero_grad()
    with torch.no_grad():
        y_fit = fitfunc(x,*params)
        loss = lossfunc(y,y_fit,weights,invsigma)
        
    fit_status["fval"] = loss.item()

    if output_sigma:
        if len(params) == 1:
            hessian = torch.autograd.functional.hessian(lambda a:lossfunc(y,fitfunc(x,a),weights,invsigma),params[0])
        else:
            hessian = torch.autograd.functional.hessian(lambda *a:lossfunc(y,fitfunc(x,*a),weights,invsigma),tuple(params))
            hessian = torch.stack([torch.stack([hessian[j][i] for i in range(len(params))]) for j in range(len(params))],1)
    
        try:
            pcov = 2*hessian.detach().pinverse()
        except(RuntimeError):
            pcov = None
    else:
        pcov = None
        
    if pcov is None:
        pcov = torch.full([nparams,nparams],np.nan)
    if not absolute_sigma:
        if y.size(0) > nparams:
            chisq = loss / (y.size(0) - nparams)
            pcov = pcov * chisq
        else:
            pcov *= np.inf
    if full_output:
        return params,pcov,optimizer.state,fit_status
    return params,pcov

def curve_fit_BS(x,y,fitfunc,init_params,sigma0=1,weights=None,nbootstrap=50,bootstrap_options={},fitter_options={},device=None,fitter_name='Pytorch_LBFGS'):

    weights_idx=[]
    fitted_params = []
    errors=[]
    chisq=[]
    chisq_transformed=[]
    niter=[]
    valid_min=[]

    npoints = y.shape[0]

    if weights is None:
        weights = bootstrap_weights(nbootstrap,npoints).to(device=device)

    if not torch.is_tensor(weights):
        weights = torch.from_numpy(weights).to(device=device)

    if not torch.is_tensor(x):
        x = torch.from_numpy(x).to(device=device)
        y = torch.from_numpy(y).to(device=device)

    for i in range(nbootstrap):
        p0 = init_params[i]
        if torch.is_tensor(p0):
            p_init = [p0.clone().detach().to(device=device).requires_grad_(True)]
        else:
            p_init = [torch.tensor(i,requires_grad=True,device=device) for i in p0]
        p,q,fitter_state,fit_status = curve_fit(fitfunc,x,y,p_init,weights=weights[i],sigma=sigma0,**fitter_options,full_output=True)
        if fit_status["is_valid"]:
            fitted_params.append(np.hstack([j.detach().cpu().numpy() for j in p]))
            errors.append(np.sqrt(np.diag(q.cpu().numpy())))
            with torch.no_grad():
                y_fit = fitfunc(x,*p)
                loss = torch.sum(((y-y_fit)/sigma0)**2)
                loss_transformed = fit_status["fval"]
            chisq.append(loss.cpu().numpy()/npoints)
            chisq_transformed.append(loss_transformed/npoints)
            valid_min.append(True)
        else:
            fitted_params.append(np.hstack([j.detach().cpu().numpy() for j in p])+np.nan)
            errors.append(np.sqrt(np.diag(q.cpu().numpy()))+np.nan)
            chisq.append(np.nan) 
            chisq_transformed.append(np.nan)
            valid_min.append(False)
        weights_idx.append(i)
        niter.append(fitter_state[p[0]]["func_evals"])

    params = np.stack(fitted_params)
        
    if 'chisq_cut' in bootstrap_options:
        chisq_median = np.median(chisq)
        masked_params = params[np.all([chisq<chisq_median*bootstrap_options["chisq_cut"],np.array(valid_min)],0)]
        mean = np.mean(masked_params,0)
        median = np.median(masked_params,0)
        std = np.std(masked_params,0)        
    else:
        mean = np.mean(params,0)
        median = np.median(params,0)
        std = np.std(params,0)

    df = create_benchmark_df(fitter_name,fitted_params,errors,npoints,weights_idx,chisq,chisq_transformed,valid_min,niter)
    return df,mean,median,std,weights

def create_benchmark_df(optimizers,params,covs,npoints,idx,chisq,chisq_transformed,is_valid,niter):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'fitter_name':optimizers,'number_points':npoints,'weights_idx':idx,'chisq':chisq,'chisq_transformed':chisq_transformed,'is_valid':is_valid,'n_iter':niter}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_weights(nfits,npoints):
    return torch.stack([torch.bincount(torch.randint(0,npoints,[npoints]),minlength=npoints) for i in range(nfits)])
