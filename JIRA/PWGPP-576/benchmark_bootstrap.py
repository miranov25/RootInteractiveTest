# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:56:37 2020

@author: majoi
"""

import numpy as np
import tensorflow as tf
import data
from FitterBFGS import bfgsfitter
import scipy.optimize
from iminuit import Minuit
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import fitter_torch

npoints = 10000
pointlist = [1000, 10000]
nfits = 20
nbootstrap = 20

sigma0=.1
sigma_initial_guess=[.2,.1]

data_sin = data.testdata()
data_sin.setfuncsin()
data_exp = data.testdata()
data_exp.setfuncexp()
data_lin = data.testdata()
data_lin.setfunclin()

def cuda_curve_fit_sync(*args, **kwargs):
    x = fitter_torch.curve_fit(*args, **kwargs)
    torch.cuda.synchronize()
    return x

def benchmark_lin():

    frames = []
    for idx, el in enumerate(pointlist):
        for ifit in range(nfits):
            data_lin.setxy(el,sigma0)
            p0 = np.random.normal(data_lin.params,sigma_initial_guess,[nfits,2])
            fitter = bfgsfitter(data.testfunc_lin_np)
            t0 = time.time()
            df0 = fitter.curve_fit_BS(data_lin.x, data_lin.y,init_params=p0,sigma0=sigma0,nbootstrap=nbootstrap)
            t1 = time.time()
            frames.append(df0)
            df0["fit_idx"] = ifit + nfits*idx
            df0["time"] = (t1-t0)/nbootstrap
            for a,b in enumerate(data_lin.params):
                df0[str.format("params_true_{}",a)]=b
            t0 = time.time()
            df0,weights=bootstrap_scipy(data_lin.x, data_lin.y,data.testfunc_lin_np,init_params=p0,sigma0=sigma0,nbootstrap=nbootstrap)
            t1 = time.time()
            df0["fit_idx"] = ifit + nfits*idx
            df0["time"] = (t1-t0)/nbootstrap
            for a,b in enumerate(data_lin.params):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
            t0 = time.time()
            df0,weights=fitter_torch.curve_fit_BS(data_lin.x, data_lin.y,data.testfunc_lin_torch,init_params=torch.from_numpy(p0),weights=weights,sigma0=sigma0,nbootstrap=nbootstrap)
            t1 = time.time()
            df0["fit_idx"] = ifit + nfits*idx
            df0["time"] = (t1-t0)/nbootstrap
            for a,b in enumerate(data_lin.params):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
    df = pd.concat(frames)
    return df

def bootstrap_weights(nfits,npoints):
    return np.stack([np.bincount(np.random.randint(0,npoints,npoints),minlength=npoints) for i in range(nfits)])

def create_benchmark_df(optimizers,params,covs,npoints,idx,chisq,chisq_transformed):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'optimizers':optimizers,'number_points':npoints,'weights_idx':idx,'chisq':chisq,'chisq_transformed':chisq_transformed}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_scipy(x,y,fitfunc,init_params,sigma0=1,weights=None,nbootstrap=50,fitter_options={},fitter_name='Scipy_LM'):
    chisq = []
    chisq_transformed=[]
    weights_idx=[]
    fitted_params = []
    errors=[]

    n = y.shape[0]
    
    if weights is None:
        weights = bootstrap_weights(nbootstrap,n)
    
    for i in range(nbootstrap):
        p,q = scipy.optimize.curve_fit(fitfunc,x,y,sigma=sigma0/np.sqrt(weights[i]),p0=init_params[i],**fitter_options)
        fitted_params.append(p)
        errors.append(np.sqrt(np.diag(q)))
        weights_idx.append(i)
        chisq.append(np.sum(((data.testfunc_lin_np(data_lin.x,*p)-data_lin.y)/sigma0)**2))
        chisq_transformed.append(np.sum(weights[i]*((data.testfunc_lin_np(data_lin.x,*p)-data_lin.y)/sigma0)**2))
        
    df = create_benchmark_df(fitter_name,fitted_params,errors,n,weights_idx,chisq,chisq_transformed)
    return df,weights

df = benchmark_lin()

df.to_pickle("benchmark_linear.pkl")