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
import fitter_minuit

np.random.seed(72654126)

npoints = 10000
pointlist = [1000,10000,100000]
nfits = 50
nbootstrap = 20

sigma0=.1
sigma_initial_guess=[.2,.1]

data_sin = data.testdata()
data_sin.setfuncsin()
data_exp = data.testdata()
data_exp.setfuncexp()
data_lin = data.testdata()
data_lin.setfunclin()

#fitters={"Scipy_LM","Pytorch_LBFGS","Pytorch_LBFGS_CUDA","iminuit","Tensorflow_BFGS"}
fitters={"Scipy_LM","Pytorch_LBFGS","iminuit"}

def cuda_curve_fit_sync(*args, **kwargs):
    x = fitter_torch.curve_fit(*args, **kwargs)
    torch.cuda.synchronize()
    return x

def benchmark_bootstrap(npoints,nfits,nbootstrap,testfunc,sigma_data,sigma_initial_guess,generate_params,nparams,xmin=-1,xmax=1,weights=None,testfunc_tf=None,testfunc_torch=None):
    frames = []
    params = []
    errors = []
    params_true = []
    fit_idx = []
    fitter_name = []
    bs_mean = []
    bs_median = []
    bs_std = []
    chisq = []
    number_points = []
    t = []
    
    if weights is None:
        weights = bootstrap_weights(nbootstrap,npoints)
    if testfunc_tf is None:
        testfunc_tf = testfunc
    if testfunc_torch is None:
        testfunc_torch = testfunc
    x = np.linspace(xmin,xmax,npoints).astype(np.float32)
    fitterTF = bfgsfitter(testfunc_tf)
    for ifit in range(nfits):
        print("Fit ",ifit)
        params_true_0 = generate_params()
        y = np.random.normal(testfunc(x,*params_true_0),sigma_data).astype(np.float32)
        p0 = np.random.normal(params_true_0,sigma_initial_guess,[nbootstrap,nparams]).astype(np.float32)

        if "Tensorflow_BFGS" in fitters:
            p,q = fitterTF.curve_fit(x,y,initial_parameters=p0[0],weights=1/sigma_data**2)
            #print(p.numpy()); print(q.numpy())
            pn = p.numpy()
            params.append(pn)
            errors.append(np.sqrt(np.diag(q.numpy())))
            chisq.append(np.sum((testfunc(x,*pn)-y)**2)/sigma_data**2/(npoints-nparams))
            params_true.append(params_true_0)
            number_points.append(npoints)
            fit_idx.append(ifit)
            fitter_name.append("Tensorflow_BFGS")
            t0 = time.time()
            df0,mean,median,std,_ = fitterTF.curve_fit_BS(x, y,weights=weights,init_params=p0,sigma0=sigma_data,nbootstrap=nbootstrap)
            t1 = time.time()
            frames.append(df0)
            df0["fit_idx"] = ifit
            df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(params_true_0):
                df0[str.format("params_true_{}",a)]=b
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)

        if "Scipy_LM" in fitters:
            try:
                p, q = scipy.optimize.curve_fit(testfunc, x, y,sigma=sigma_data*np.ones_like(y),p0=p0[0])
            except(RuntimeError):
                p = np.full(nparams,np.nan)
                q = np.full([nparams,nparams],np.nan)
            #print(p); print(q)
            params.append(p)
            errors.append(np.sqrt(np.diag(q)))
            chisq.append(np.sum((testfunc(x,*p)-y)**2)/sigma_data**2/(npoints-nparams))
            params_true.append(params_true_0)
            number_points.append(npoints)
            fit_idx.append(ifit)
            fitter_name.append("Scipy_LM")
            t0 = time.time()
            df0,mean,median,std,_=bootstrap_scipy(x, y,testfunc,init_params=p0,weights=weights,sigma0=sigma_data,nbootstrap=nbootstrap,bootstrap_options={'chisq_cut':10})
            t1 = time.time()
            df0["fit_idx"] = ifit
            df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(params_true_0):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)

        if "Pytorch_LBFGS" in fitters:
            p,q,optim_state,fit_status = fitter_torch.curve_fit(testfunc_torch,torch.from_numpy(x),torch.from_numpy(y),[torch.tensor(i,requires_grad=True) for i in p0[0]],sigma=sigma_data,full_output=True)
            #print(p[0].detach().numpy()); print(q.numpy())
            if fit_status["is_valid"]:
                pn = np.hstack([j.detach().cpu().numpy() for j in p])
                params.append(pn)
                errors.append(np.sqrt(np.diag(q.cpu().numpy())))
                chisq.append(fit_status['fval']/(npoints-nparams))
            else:
                params.append(np.full(nparams,np.nan))
                errors.append(np.full(nparams,np.nan))
                chisq.append(np.nan)
            params_true.append(params_true_0)
            number_points.append(npoints)
            fit_idx.append(ifit)
            fitter_name.append("Pytorch_LBFGS")
            t0 = time.time()
            df0,mean,median,std,_=fitter_torch.curve_fit_BS(x, y,testfunc_torch,init_params=p0,weights=weights,sigma0=sigma_data,nbootstrap=nbootstrap,bootstrap_options={'chisq_cut':10})
            t1 = time.time()
            df0["fit_idx"] = ifit
            df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(params_true_0):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)


        if torch.cuda.is_available() and "Pytorch_LBFGS_CUDA" in fitters:
            p,q,optim_state,fit_status = fitter_torch.curve_fit(testfunc_torch,torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda(),[torch.tensor(i,requires_grad=True,device="cuda:0") for i in p0[0]],sigma=sigma_data,full_output=True)
            #print(p[0].detach().numpy()); print(q.numpy())
            if fit_status["is_valid"]:
                pn = np.hstack([j.detach().cpu().numpy() for j in p])
                params.append(pn)
                errors.append(np.sqrt(np.diag(q.cpu().numpy())))
                chisq.append(fit_status['fval']/(npoints-nparams))
            else:
                params.append(np.full(nparams,np.nan))
                errors.append(np.full(nparams,np.nan))
                chisq.append(np.nan)
            params_true.append(params_true_0)
            number_points.append(npoints)
            fit_idx.append(ifit)
            fitter_name.append("Pytorch_LBFGS_CUDA")
            t0 = time.time()
            df0,mean,median,std,_=fitter_torch.curve_fit_BS(x, y,testfunc_torch,init_params=p0,weights=weights,sigma0=sigma_data,nbootstrap=nbootstrap,device="cuda:0",fitter_name="Pytorch_LBFGS_CUDA",bootstrap_options={'chisq_cut':10})
            torch.cuda.synchronize()
            t1 = time.time()
            df0["fit_idx"] = ifit
            df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(params_true_0):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)
            
        if "iminuit" in fitters:
            p,q,status = fitter_minuit.curve_fit(testfunc, x, y,weights=1/sigma_data**2,p0=p0[0],full_output=True)
            #print(p[0].detach().numpy()); print(q.numpy())
            if status.is_valid:
                params.append(p)
                errors.append(np.sqrt(np.diag(q)))
                chisq.append(status.fval/(npoints-nparams))
            else:
                params.append(np.full(nparams,np.nan))
                errors.append(np.full(nparams,np.nan))
                chisq.append(np.nan)
            params_true.append(params_true_0)
            number_points.append(npoints)
            fit_idx.append(ifit)
            fitter_name.append("Minuit")
            t0 = time.time()
            df0,mean,median,std,_=fitter_minuit.curve_fit_BS(x, y,testfunc,init_params=p0,weights=weights,sigma0=sigma_data,nbootstrap=nbootstrap,bootstrap_options={'chisq_cut':10})
            t1 = time.time()
            df0["fit_idx"] = ifit
            df0["time"] = (t1-t0)/nbootstrap
            t.append(t1-t0)
            for a,b in enumerate(params_true_0):
                df0[str.format("params_true_{}",a)]=b
            frames.append(df0)
            bs_mean.append(mean)
            bs_median.append(median)
            bs_std.append(std)        

    bs_mean = np.stack(bs_mean)
    bs_median = np.stack(bs_median)
    bs_std = np.stack(bs_std)
    params = np.stack(params)
    errors = np.stack(errors)
    params_true = list(zip(*params_true))
    d = {"fitter_name":fitter_name,"fit_idx":fit_idx,"number_points":number_points,"time":t,"chisq":chisq,"nbootstrap":nbootstrap}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):errors[:,i] for i in range(errors.shape[1])})
    d.update({str.format("bs_mean_{}",i):bs_mean[:,i] for i in range(bs_mean.shape[1])})
    d.update({str.format("bs_median_{}",i):bs_median[:,i] for i in range(bs_median.shape[1])})
    d.update({str.format("bs_std_{}",i):bs_std[:,i] for i in range(bs_std.shape[1])})
    d.update({str.format("params_true_{}",idx):el for idx,el in enumerate(params_true)})

    df1 = pd.DataFrame(d)
    df2 = pd.concat(frames)

    return df1,df2

def bootstrap_weights(nfits,npoints):
    return np.stack([np.bincount(np.random.randint(0,npoints,npoints),minlength=npoints) for i in range(nfits)])

def create_benchmark_df(optimizers,params,covs,npoints,idx,chisq,chisq_transformed,niter):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'fitter_name':optimizers,'number_points':npoints,'weights_idx':idx,'chisq':chisq,'chisq_transformed':chisq_transformed,'n_iter':niter}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_scipy(x,y,fitfunc,init_params,sigma0=1,weights=None,nbootstrap=50,bootstrap_options={},fitter_options={},fitter_name='Scipy_LM'):
    chisq = []
    chisq_transformed=[]
    weights_idx=[]
    fitted_params = []
    errors=[]
    niter=[]
    is_valid=[]

    n = y.shape[0]
    nparams = init_params.shape[0]

    if weights is None:
        weights = bootstrap_weights(nbootstrap,n)

    for i in range(nbootstrap):
        try:
            p, q, infodict, errmsg, ier = scipy.optimize.curve_fit(fitfunc,x,y,sigma=sigma0/np.sqrt(weights[i]),p0=init_params[i],**fitter_options, full_output=True)
        except(RuntimeError):
            p = np.zeros_like(init_params[i])+np.nan
            fitted_params.append(p)
            errors.append(p)
            weights_idx.append(i)
            chisq.append(np.nan)
            chisq_transformed.append(np.nan)
            niter.append(np.nan)
            is_valid.append(False)
            continue
        fitted_params.append(p)
        errors.append(np.sqrt(np.diag(q)))
        weights_idx.append(i)
        chisq.append(np.sum(((fitfunc(x,*p)-y)/sigma0)**2)/(n-nparams))
        chisq_transformed.append(np.sum(weights[i]*((fitfunc(x,*p)-y)/sigma0)**2)/(n-nparams))
        niter.append(infodict['nfev'])
        is_valid.append(True)

    df = create_benchmark_df(fitter_name,fitted_params,errors,n,weights_idx,chisq,chisq_transformed,niter)
    params = np.stack(fitted_params)
    if 'chisq_cut' in bootstrap_options:
        chisq_median = np.median(chisq)
        masked_params = params[np.all([chisq<chisq_median*bootstrap_options["chisq_cut"],np.array(is_valid)],0)]
        mean = np.mean(masked_params,0)
        median = np.median(masked_params,0)
        std = np.std(masked_params,0)        
    else:
        mean = np.nanmean(params,0)
        median = np.nanmedian(params,0)
        std = np.nanstd(params,0)
    return df,mean,median,std,weights

def apply_test(test,df,alarmsigma=3,to_markdown=True):
    g = df.groupby(["number_points","fitter_name"])
    r = g.apply(lambda x:test(x,alarmsigma))
    if to_markdown:
        print(r.to_markdown())
    else:
        print(r)

def test_mean(group,alarmsigma=3):
    return pd.Series({
            'mean_0':group['delta_0'].mean(),
            'bootstrap_mean_0':(group['params_true_0']-group['bs_mean_0']).mean(),
            'rmsexp_0':np.sqrt((group["errors_0"]**2).mean())/np.sqrt(len(group.index)),
            'status_0':np.abs(group['delta_0'].mean()) < alarmsigma * np.sqrt((group["errors_0"]**2).mean())/np.sqrt(len(group.index))
            })

def test_rms(group,alarmsigma=3):
    return pd.Series({
            'std_0':group["delta_0"].std(),
            'bootstrap_std_0':group["bs_std_0"].mean(),
            'rms_estimate_0':np.sqrt((group["errors_0"]**2).mean()),
            'status_0':np.abs(group["delta_0"].std()-np.sqrt((group["errors_0"]**2).mean()))<alarmsigma*np.sqrt((group["errors_0"]**2).mean())/np.sqrt(len(group.index))
            })

def test_pull(group,alarmsigma=3):
    return pd.Series({
            'pull_mean_0':group["pull_0"].mean(),
            'pull_std_0':group["pull_0"].std(),
            'status_0':np.abs(group["pull_0"].mean())<alarmsigma/np.sqrt(len(group.index)) and np.abs(group["pull_0"].std()-1)<alarmsigma/np.sqrt(len(group.index))
            })

def test_chisq(group,alarmsigma=3):
    return pd.Series({
            'chisq_mean':group["chisq"].mean(),
            'chisq_std': group["chisq"].std(),
            'status':np.abs(group["chisq"].mean()-1)<alarmsigma/np.sqrt(len(group.index))
            })    
    """
print("Linear: ")
df1s = []
df2s = []
for idx, el in enumerate(pointlist):
    df1,df2 = benchmark_bootstrap(el,nfits,nbootstrap,data.testfunc_lin_np,sigma0,.4,lambda:-np.random.rand(2).astype(np.float32),2)
    df1s.append(df1)
    df2s.append(df2)
df1 = pd.concat(df1s)
df2 = pd.concat(df2s)
print("Linear: ")
df1["delta_0"] = df1["params_true_0"] - df1["params_0"]
df1["delta_1"] = df1["params_true_1"] - df1["params_1"]
df1["pull_0"] = df1["delta_0"] / df1["errors_0"]
df1["pull_1"] = df1["delta_1"] / df1["errors_1"]
apply_test(test_mean,df1)
apply_test(test_rms,df1)
apply_test(test_pull,df1)
apply_test(test_chisq,df1)
print(df2.groupby(["fitter_name","number_points"]).mean()[["time","n_iter"]].to_markdown())

df2.to_pickle("benchmark_linear_eachfit.pkl")
df1.to_pickle("benchmark_linear_bootstrap.pkl")

print("Exponential: ")
df1s = []
df2s = []
for idx, el in enumerate(pointlist):
    df1,df2 = benchmark_bootstrap(el,nfits,nbootstrap,data.testfunc_exp_np,sigma0,.4,lambda:-np.random.rand(2).astype(np.float32),2,testfunc_torch=data.testfunc_exp_torch,testfunc_tf=data.testfunc_exp)
    df1s.append(df1)
    df2s.append(df2)
df1 = pd.concat(df1s)
df2 = pd.concat(df2s)
print("Exponential: ")
df1["delta_0"] = df1["params_true_0"] - df1["params_0"]
df1["delta_1"] = df1["params_true_1"] - df1["params_1"]
df1["pull_0"] = df1["delta_0"] / df1["errors_0"]
df1["pull_1"] = df1["delta_1"] / df1["errors_1"]
apply_test(test_mean,df1)
apply_test(test_rms,df1)
apply_test(test_pull,df1)
apply_test(test_chisq,df1)

print(df2.groupby(["fitter_name","number_points"]).mean()[["time","n_iter"]].to_markdown())

df2.to_pickle("benchmark_exponential_eachfit.pkl")
df1.to_pickle("benchmark_exponential_bootstrap.pkl")
"""
print("Gaussian: ")
df1s = []
df2s = []
for idx, el in enumerate(pointlist):
    df1,df2 = benchmark_bootstrap(el,nfits,nbootstrap,data.testfunc_gaus_np,sigma0,.4,lambda:np.array([np.random.rand()+.5,np.random.rand()-.5,np.random.rand()*3]).astype(np.float32),3,testfunc_torch=data.testfunc_gaus_torch,testfunc_tf=data.testfunc_gaus_tf,xmin=-3,xmax=3)
    df1s.append(df1)
    df2s.append(df2)
df1 = pd.concat(df1s)
df2 = pd.concat(df2s)
print("Gaussian: ")
df1["delta_0"] = df1["params_true_0"] - df1["params_0"]
df1["delta_1"] = df1["params_true_1"] - df1["params_1"]
df1["delta_2"] = df1["params_true_2"] - df1["params_2"]
df1["pull_0"] = df1["delta_0"] / df1["errors_0"]
df1["pull_1"] = df1["delta_1"] / df1["errors_1"]
df1["pull_2"] = df1["delta_2"] / df1["errors_2"]
apply_test(test_mean,df1)
apply_test(test_rms,df1)
apply_test(test_pull,df1)
apply_test(test_chisq,df1)

print(df2.groupby(["fitter_name","number_points"]).mean()[["time","n_iter"]].to_markdown())

df2.to_pickle("benchmark_gaussian_eachfit.pkl")
df1.to_pickle("benchmark_gaussian_bootstrap.pkl")