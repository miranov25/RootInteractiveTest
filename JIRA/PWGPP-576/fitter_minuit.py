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
    if not absolute_sigma:
        if y.size > p0.size:
            chisq = m.fval / (y.size - p0.size)
            cov = cov * chisq
        else:
            cov *= np.inf
    if full_output:
        return m.np_values(),cov,status
    return m.np_values(),cov


def curve_fit_BS(x,y,fitfunc,init_params,sigma0=1,weights=None,nbootstrap=50,bootstrap_options={},fitter_options={},fitter_name='Minuit'):
    chisq = []
    chisq_transformed=[]
    weights_idx=[]
    fitted_params = []
    errors=[]
    niter=[]
    valid_min=[]

    n = y.shape[0]
    nparams = init_params.shape[0]

    if weights is None:
        weights = bootstrap_weights(nbootstrap,n)

    for i in range(nbootstrap):
        p, q, status = curve_fit(fitfunc,x,y,weights=weights[i]/sigma0**2,p0=init_params[i],**fitter_options, full_output=True)
        if status.is_valid:
            fitted_params.append(p)
            errors.append(np.sqrt(np.diag(q)))
            chisq.append(np.sum(((fitfunc(x,*p)-y)/sigma0)**2)/(n-nparams))
            chisq_transformed.append(status.fval/(n-nparams))
        else:
            fitted_params.append(p+np.nan)
            errors.append(np.sqrt(np.diag(q))+np.nan)
            chisq.append(np.nan) 
            chisq_transformed.append(np.nan)
        weights_idx.append(i)
        niter.append(status.nfcn)
        valid_min.append(status.is_valid)


    params = np.stack(fitted_params)
    if 'chisq_cut' in bootstrap_options:
        chisq_median = np.nanmedian(chisq)
        is_accepted = np.all([chisq<chisq_median*bootstrap_options["chisq_cut"],np.array(valid_min)],0)
        masked_params = params[is_accepted]
        mean = np.mean(masked_params,0)
        median = np.median(masked_params,0)
        std = np.std(masked_params,0)        
    else:
        mean = np.nanmean(params,0)
        median = np.nanmedian(params,0)
        std = np.nanstd(params,0)
        is_accepted = True
    df = create_benchmark_df(fitter_name,fitted_params,errors,n,weights_idx,chisq,chisq_transformed,valid_min,is_accepted,niter)
    return df,mean,median,std,weights    

def create_benchmark_df(optimizers,params,covs,npoints,idx,chisq,chisq_transformed,is_valid,is_accepted,niter):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'fitter_name':optimizers,'number_points':npoints,'weights_idx':idx,'chisq':chisq,'chisq_transformed':chisq_transformed,'is_valid':is_valid,'is_accepted':is_accepted,'n_iter':niter}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_weights(nfits,npoints):
    return np.stack([np.bincount(np.random.randint(0,npoints,[npoints]),minlength=npoints) for i in range(nfits)])
