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

np.random.seed(72654126)

npoints = 10000
pointlist = [1000, 10000]
nfits = 50
nbootstrap = 50

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
    params = []
    errors = []
    params_true = []
    fit_idx = []
    fitter_name = []
    bs_mean = []
    bs_median = []
    bs_std = []
    number_points = []
    t = []
    for idx, el in enumerate(pointlist):
        fitterTF = bfgsfitter(data.testfunc_lin_np)
        weights = bootstrap_weights(nbootstrap,el).astype(np.float32)
        for ifit in range(nfits):
            print("idx:", idx, "Fit ",ifit)
            data_lin.setxy(el,sigma0)
            p0 = np.random.normal(data_lin.params,sigma_initial_guess,[nfits,2]).astype(np.float32)
            p,q = fitterTF.curve_fit(data_lin.x,data_lin.y,initial_parameters=p0[0],weights=1/sigma0**2)
            #print(p.numpy()); print(q.numpy())
            params.append(p.numpy())
            errors.append(np.sqrt(np.diag(q.numpy())))
            params_true.append(data_lin.params)
            number_points.append(el)
            fit_idx.append(ifit + nfits*idx)
            fitter_name.append("Tensorflow_BFGS")
            t0 = time.time()
            df0,mean,median,std,_ = fitterTF.curve_fit_BS(data_lin.x, data_lin.y,weights=weights,init_params=p0,sigma0=sigma0,nbootstrap=nbootstrap)
            t1 = time.time()
            frames.append(df0)
            df0["fit_idx"] = ifit + nfits*idx
            #df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(data_lin.params):
                df0[str.format("params_true_{}",a)]=b
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)
            
            p, q = scipy.optimize.curve_fit(data.testfunc_lin_np, data_lin.x, data_lin.y,sigma=sigma0*np.ones_like(data_lin.y),p0=p0[0])
            #print(p); print(q)
            params.append(p)
            errors.append(np.sqrt(np.diag(q)))
            params_true.append(data_lin.params)
            number_points.append(el)
            fit_idx.append(ifit + nfits*idx)
            fitter_name.append("Scipy_LM")
            t0 = time.time()
            df0,mean,median,std,_=bootstrap_scipy(data_lin.x, data_lin.y,data.testfunc_lin_np,init_params=p0,weights=weights,sigma0=sigma0,nbootstrap=nbootstrap)
            t1 = time.time()
            df0["fit_idx"] = ifit + nfits*idx
            #df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(data_lin.params):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)
            
            p,q = fitter_torch.curve_fit(data.testfunc_lin_torch,torch.from_numpy(data_lin.x),torch.from_numpy(data_lin.y),[torch.tensor(p0[0],requires_grad=True)],sigma=sigma0)
            #print(p[0].detach().numpy()); print(q.numpy())
            params.append(np.hstack([j.detach().numpy() for j in p]))
            errors.append(np.sqrt(np.diag(q.numpy())))
            params_true.append(data_lin.params)
            number_points.append(el)
            fit_idx.append(ifit + nfits*idx)
            fitter_name.append("Pytorch_LBFGS")
            t0 = time.time()
            df0,mean,median,std,_=fitter_torch.curve_fit_BS(data_lin.x, data_lin.y,data.testfunc_lin_np,init_params=p0,weights=weights,sigma0=sigma0,nbootstrap=nbootstrap)
            t1 = time.time()
            df0["fit_idx"] = ifit + nfits*idx
            #df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(data_lin.params):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)
    df = pd.concat(frames)
    bs_mean = np.stack(bs_mean)
    bs_median = np.stack(bs_median)
    bs_std = np.stack(bs_std)
    params = np.stack(params)
    errors = np.stack(errors)
    params_true = list(zip(*params_true))
    d = {"fitter_name":fitter_name,"fit_idx":fit_idx,"number_points":number_points,"time":t,"nbootstrap":nbootstrap}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):errors[:,i] for i in range(errors.shape[1])})
    d.update({str.format("bs_mean_{}",i):bs_mean[:,i] for i in range(bs_mean.shape[1])})
    d.update({str.format("bs_median_{}",i):bs_median[:,i] for i in range(bs_median.shape[1])})
    d.update({str.format("bs_std_{}",i):bs_std[:,i] for i in range(bs_std.shape[1])})
    d.update({str.format("params_true_{}",idx):el for idx,el in enumerate(params_true)})
    
    df1 = pd.DataFrame(d)
    return df,df1

def benchmark_bootstrap(npoints,nfits,nbootstrap,testfunc,sigma_data,sigma_initial_guess,generate_params,xmin=-1,xmax=1,weights=None):
    frames = []
    params = []
    errors = []
    params_true = []
    fit_idx = []
    fitter_name = []
    bs_mean = []
    bs_median = []
    bs_std = []
    number_points = []
    t = []
    if weights is None:
        weights = bootstrap_weights(nbootstrap,npoints)
    x = np.linspace(xmin,xmax,npoints)
    for ifit in range(nfits):
        params = generate_params()
        y = np.random.normal(testfunc(x,*params),sigma_data)
        p0 = np.random.normal(data_lin.params,sigma_initial_guess,[nfits,2])
        fitter = bfgsfitter(testfunc)
        t0 = time.time()
        df0,mean,median,std,_ = fitter.curve_fit_BS(x, y,init_params=p0,weights=weights,sigma0=sigma0,nbootstrap=nbootstrap)
        t1 = time.time()
        frames.append(df0)
        df0["fit_idx"] = ifit
        df0["time"] = (t1-t0)/nbootstrap
        for a,b in enumerate(params):
            df0[str.format("params_true_{}",a)]=b
        t0 = time.time()
        df0,mean,median,std,_=bootstrap_scipy(x, y,testfunc,init_params=p0,weights=weights,sigma0=sigma0,nbootstrap=nbootstrap)
        t1 = time.time()
        df0["fit_idx"] = ifit
        df0["time"] = (t1-t0)/nbootstrap
        for a,b in enumerate(params):
            df0[str.format("params_true_{}",a)]=b
        frames.append(df0)
        t0 = time.time()
        df0,mean,median,std,_=fitter_torch.curve_fit_BS(x, y,testfunc,init_params=torch.from_numpy(p0),weights=weights,sigma0=sigma0,nbootstrap=nbootstrap)
        t1 = time.time()
        df0["fit_idx"] = ifit
        df0["time"] = (t1-t0)/nbootstrap
        for a,b in enumerate(params):
            df0[str.format("params_true_{}",a)]=b
        frames.append(df0)
    df2 = pd.concat(frames)
    return df1,df2    

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
    params = np.stack(fitted_params)
    mean = np.mean(params,0)
    median = np.median(params,0)
    std = np.std(params,0)
    return df,mean,median,std,weights

df2,df1 = benchmark_lin()

df2.to_pickle("benchmark_linear_eachfit.pkl")
df1.to_pickle("benchmark_linear_bootstrap.pkl")

df1["delta_0"] = df1["params_true_0"] - df1["params_0"]
df1["delta_1"] = df1["params_true_1"] - df1["params_1"]
df1["pull_0"] = df1["delta_0"] / df1["errors_0"]
df1["pull_1"] = df1["delta_1"] / df1["errors_1"]
#
df1_tf = df1.query("fitter_name=='Tensorflow_BFGS'")
df1_scipy= df1.query("fitter_name=='Scipy_LM'")
df1_torch= df1.query("fitter_name=='Pytorch_LBFGS'")
N = len(df1_tf.index)

print("Fit type:\tmean\t\trms_estimate\t\tstatus")
print("Tensorflow:\t%8.6F\t%8.6F\t%8.0F" % (df1_tf["delta_0"].mean(), df1_tf["errors_0"].mean()/np.sqrt(N), \
      np.abs(df1_tf["delta_0"].mean())< 3* df1_tf["errors_0"].mean()/np.sqrt(N)))
print("Scipy:\t%8.6F\t%8.6F\t%8.0F" % (df1_scipy["delta_0"].mean(), df1_scipy["errors_0"].mean()/np.sqrt(N), \
      np.abs(df1_scipy["delta_0"].mean())< 3* df1_scipy["errors_0"].mean()/np.sqrt(N)))
print("Pytorch:\t%8.6F\t%8.6F\t%8.0F" % (df1_torch["delta_0"].mean(), df1_torch["errors_0"].mean()/np.sqrt(N), \
      np.abs(df1_torch["delta_0"].mean())< 3* df1_torch["errors_0"].mean()/np.sqrt(N)))
print("Tensorflow:\t%8.6F\t%8.6F\t%8.0F" % (df1_tf["delta_1"].mean(), df1_tf["errors_1"].mean()/np.sqrt(N), \
      np.abs(df1_tf["delta_1"].mean())< 3* df1_tf["errors_1"].mean()/np.sqrt(N)))
print("Scipyi:\t%8.6F\t%8.6F\t%8.0F" % (df1_scipy["delta_1"].mean(), df1_scipy["errors_1"].mean()/np.sqrt(N), \
      np.abs(df1_scipy["delta_1"].mean())< 3* df1_scipy["errors_1"].mean()/np.sqrt(N)))
print("Pytorch:\t%8.6F\t%8.6F\t%8.0F" % (df1_torch["delta_1"].mean(), df1_torch["errors_1"].mean()/np.sqrt(N), \
      np.abs(df1_torch["delta_1"].mean())< 3* df1_torch["errors_1"].mean()/np.sqrt(N)))

print("Fit type:\tstd\t\tbootstrap_std\t\trms_estimate\t status")
isOK=np.abs(df1_tf["delta_0"].std()-df1_tf["errors_0"].mean())<3*df1_tf["errors_0"].mean()/np.sqrt(N)
print("Tensorflow:\t%8.6F\t%8.6F\t\t%8.6F\t%2.0F" % (df1_tf["delta_0"].std(),df1_tf["bs_std_0"].mean(), df1_tf["errors_0"].mean(),isOK))

isOK=np.abs(df1_scipy["delta_0"].std()-df1_scipy["errors_0"].mean())<3*df1_scipy["errors_0"].mean()/np.sqrt(N)
print("Scipy:\t\t%8.6F\t%8.6F\t\t%8.6F\t%2.0F" % (df1_scipy["delta_0"].std(),df1_scipy["bs_std_0"].mean(), df1_scipy["errors_0"].mean(),isOK))

isOK=np.abs(df1_torch["delta_0"].std()-df1_torch["errors_0"].mean())<3*df1_torch["errors_0"].mean()/np.sqrt(N)
print("Pytorch:\t%8.6F\t%8.6F\t\t%8.6F\t%2.0F" % (df1_torch["delta_0"].std(),df1_torch["bs_std_0"].mean(), df1_torch["errors_0"].mean(),isOK))

isOK=np.abs(df1_tf["delta_1"].std()-df1_tf["errors_1"].mean())<3*df1_tf["errors_1"].mean()/np.sqrt(N)
print("Tensorflow:\t%8.6F\t%8.6F\t\t%8.6F\t%2.0F" % (df1_tf["delta_1"].std(),df1_tf["bs_std_1"].mean(), df1_tf["errors_1"].mean(),isOK))

isOK=np.abs(df1_scipy["delta_1"].std()-df1_scipy["errors_1"].mean())<3*df1_scipy["errors_1"].mean()/np.sqrt(N)
print("Scipy:\t\t%8.6F\t%8.6F\t\t%8.6F\t%2.0F" % (df1_scipy["delta_1"].std(),df1_scipy["bs_std_1"].mean(), df1_scipy["errors_1"].mean(),isOK))

isOK=np.abs(df1_torch["delta_1"].std()-df1_torch["errors_1"].mean())<3*df1_torch["errors_1"].mean()/np.sqrt(N)
print("Pytorch:\t%8.6F\t%8.6F\t\t%8.6F\t%2.0F" % (df1_torch["delta_1"].std(),df1_torch["bs_std_1"].mean(), df1_torch["errors_1"].mean(),isOK))

print("Fit type:\tpull mean\tpull std\tstatus")
isOK=np.abs(df1_tf["pull_0"].mean())<3/np.sqrt(N) and np.abs(df1_tf["pull_0"].std()-1)<3/np.sqrt(N)
print("Tensorflow:\t%8.6F\t%8.6F\t%2.0F" % (df1_tf["pull_0"].mean(),df1_tf["pull_0"].std(),isOK))

isOK=np.abs(df1_scipy["pull_0"].mean())<3/np.sqrt(N) and np.abs(df1_scipy["pull_0"].std()-1)<3/np.sqrt(N)
print("Scipy:\t\t%8.6F\t%8.6F\t%2.0F" % (df1_scipy["pull_0"].mean(),df1_scipy["pull_0"].std(),isOK))

isOK=np.abs(df1_torch["pull_0"].mean())<3/np.sqrt(N) and np.abs(df1_torch["pull_0"].std()-1)<3/np.sqrt(N)
print("Pytorch:\t%8.6F\t%8.6F\t%2.0F" % (df1_torch["pull_0"].mean(),df1_torch["pull_0"].std(),isOK))

isOK=np.abs(df1_tf["pull_1"].mean())<3/np.sqrt(N) and np.abs(df1_tf["pull_1"].std()-1)<3/np.sqrt(N)
print("Tensorflow:\t%8.6F\t%8.6F\t%2.0F" % (df1_tf["pull_1"].mean(),df1_tf["pull_1"].std(),isOK))

isOK=np.abs(df1_scipy["pull_1"].mean())<3/np.sqrt(N) and np.abs(df1_scipy["pull_1"].std()-1)<3/np.sqrt(N)
print("Scipy:\t\t%8.6F\t%8.6F\t%2.0F" % (df1_scipy["pull_1"].mean(),df1_scipy["pull_1"].std(),isOK))

isOK=np.abs(df1_torch["pull_1"].mean())<3/np.sqrt(N) and np.abs(df1_torch["pull_1"].std()-1)<3/np.sqrt(N)
print("Pytorch:\t%8.6F\t%8.6F\t%2.0F" % (df1_torch["pull_1"].mean(),df1_torch["pull_1"].std(),isOK))