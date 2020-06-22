# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:14:15 2020

@author: majoi
"""
from iminuit import Minuit
import pandas as pd
import numpy as np

def mse(x,y,weights):
    return np.sum(weights*(x-y)**2)

def curve_fit(fitfunc,x,y,p0, weights = 1, lossfunc = mse, absolute_sigma=False, initial_errors = 1, errordef = 1, optimizer_options={},full_output=False):
    def loss(params):
        y_predicted = fitfunc(x,*params)
        return lossfunc(y_predicted,y,weights)
    m = Minuit.from_array_func(loss,p0,errordef=errordef,error=initial_errors,**optimizer_options)
    status,params = m.migrad()
    
    if not status.hesse_failed:
        cov = m.np_matrix()
    else:
        cov = np.full([p0.size,p0.size],np.nan)
    
    if full_output:
        return m.values.values(),cov,status
    return m.values.values(),cov


def curve_fit_BS(x,y,fitfunc,init_params,sigma0=1,weights=None,nbootstrap=50,fitter_options={},fitter_name='Minuit'):
    chisq = []
    chisq_transformed=[]
    weights_idx=[]
    fitted_params = []
    errors=[]
    niter=[]
    valid_min=[]

    n = y.shape[0]

    if weights is None:
        weights = bootstrap_weights(nbootstrap,n)

    for i in range(nbootstrap):
        p, q, status = curve_fit(fitfunc,x,y,weights=weights[i]/sigma0**2,p0=init_params[i],**fitter_options, full_output=True)
        fitted_params.append(p)
        errors.append(np.sqrt(np.diag(q)))
        weights_idx.append(i)
        chisq.append(np.sum(((fitfunc(x,*p)-y)/sigma0)**2))
        chisq_transformed.append(np.sum(weights[i]*((fitfunc(x,*p)-y)/sigma0)**2))
        niter.append(status.nfcn)
        valid_min.append(status.is_valid)

    df = create_benchmark_df(fitter_name,fitted_params,errors,n,weights_idx,chisq,chisq_transformed,niter)
    params = np.stack(fitted_params)
    mean = np.nanmean(params,0)
    median = np.nanmedian(params,0)
    std = np.nanstd(params,0)
    return df,mean,median,std,weights    

def create_benchmark_df(optimizers,params,covs,npoints,idx,chisq,chisq_transformed,niter):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'fitter_name':optimizers,'number_points':npoints,'weights_idx':idx,'chisq':chisq,'chisq_transformed':chisq_transformed,'n_iter':niter}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_weights(nfits,npoints):
    return np.stack([np.bincount(np.random.randint(0,npoints,[npoints]),minlength=npoints) for i in range(nfits)])
