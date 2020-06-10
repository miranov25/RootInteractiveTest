# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:25:28 2020

@author: mkroe. majoi
"""
# TODO - parameterize optimizer_options and fit_options (starting from pytorch)

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

data_sin = data.testdata()
data_sin.setfuncsin()
data_exp = data.testdata()
data_exp.setfuncexp()
data_lin = data.testdata()
data_lin.setfunclin()

#pointlist = [10,100,1000,10000,100000,1000000]
#pointlist = [10, 100, 1000,10000, 100000]
pointlist = [1000, 10000]
nfits = 100
sigma0=0.1
eps=1e-3

np.random.seed(72654126)

lin_initial_guess_sigma = .4

lin_parameter_list = []
lin_parameter_list_org = []
lin_parameter_list_sigma = []
linlist = []
sinlist = []
explist = []

def cuda_curve_fit_sync(*args, **kwargs):
    x = fitter_torch.curve_fit(*args, **kwargs)
    torch.cuda.synchronize()
    return x

def bootstrap_weights(nfits,npoints):
    return np.stack([np.bincount(np.random.randint(0,npoints,npoints),minlength=npoints) for i in range(nfits)])

def create_benchmark_df(name,optimizers,params_true,params,covs,npoints,idx,chisq,chisq_transformed):
    params = np.stack(params)
    covs = np.stack(covs)
    params_true = list(zip(*params_true))
    d = {'type':name,'optimizers':optimizers,'number_points':npoints,'weights_idx':idx}
    d.update({str.format("params_true_{}",idx):el for idx,el in enumerate(params_true)})
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def benchmark_linear(pointList):

    params = []
    covs = []
    params_true = []
    optimizers = []
    npoints = []
    weights_index = []

    for idx, el in enumerate(pointList):
        comp_time_lin = []
        lin_fitter = bfgsfitter(data.testfunc_lin)
        data_lin.setxy(el,sigma0)
        lin_parameter_list_org.append(data_lin.params)
        #weights = np.random.rand(nfits,el)*2
        weights = bootstrap_weights(nfits,el)+eps
        weights[0] = np.ones(el)
        p0 = np.random.normal(data_lin.params,lin_initial_guess_sigma,[nfits,2])
        # bfgsfitter
        t1_start = time.time()
        for i in range(nfits):
            p, q = lin_fitter.curve_fit(data_lin.x, data_lin.y,weights=tf.constant(weights[i]/sigma0**2),initial_parameters=tf.cast(p0[i],tf.float32))
            weights_index.append(i)
            params.append(p.numpy())
            covs.append(np.diag(q.numpy()))
            params_true.append(data_lin.params)
            optimizers.append("Tensorflow_BFGS")
            npoints.append(el)
            chisq.append(np.sum(((data.testfunc_lin_np(data_lin.x,*p.numpy())-data_lin.y)/sigma0)**2))
            chisq_transformed.append(np.sum(weights[i]*((data.testfunc_lin_np(data_lin.x,*p.numpy())-data_lin.y)/sigma0)**2))
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(p)
        print(q)
        print(t1_stop - t1_start)


        # second fit after initializiation
        t1_start = time.time()
        for i in range(nfits):
            p, q = lin_fitter.curve_fit(data_lin.x, data_lin.y,weights=tf.constant(weights[i]/sigma0**2),initial_parameters=tf.cast(p0[i],tf.float32))
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(t1_stop - t1_start)

        # scipy
        t1_start = time.time()
        for i in range(nfits):
            p, q = scipy.optimize.curve_fit(data.testfunc_lin_np, data_lin.x, data_lin.y,sigma=sigma0/np.sqrt(weights[i]),p0=p0[i])
            params.append(p)
            covs.append(np.diag(q))
            params_true.append(data_lin.params)
            optimizers.append("Scipy_LM")
            npoints.append(el)
            weights_index.append(i)
            chisq.append(np.sum(((data.testfunc_lin_np(data_lin.x,*p)-data_lin.y)/sigma0)**2))
            chisq_transformed.append(np.sum(weights[i]*((data.testfunc_lin_np(data_lin.x,*p)-data_lin.y)/sigma0)**2))
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(p)
        print(q)
        print(t1_stop - t1_start)


        # pytorch CPU
        t1_start = time.time()
        x = torch.from_numpy(data_lin.x)
        y = torch.from_numpy(data_lin.y)
        for i in range(nfits):
            p, q = fitter_torch.curve_fit(data.testfunc_lin_torch, x, y,
                                             [torch.ones(2, requires_grad=True, dtype=torch.float64)],weights=torch.from_numpy(weights[i]/sigma0**2))

            params.append(p[0].detach().numpy())
            covs.append(np.diag(q.numpy()))
            params_true.append(data_lin.params)
            optimizers.append("PyTorch_LBFGS")
            npoints.append(el)
            weights_index.append(i)
            chisq.append(np.sum(((data.testfunc_lin_np(data_lin.x,*p[0].detach().numpy())-data_lin.y)/sigma0)**2))
            chisq_transformed.append(np.sum(weights[i]*((data.testfunc_lin_np(data_lin.x,*p[0].detach().numpy())-data_lin.y)/sigma0)**2))
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(p)
        print(q)
        print(t1_stop - t1_start)
        # pytorch GPU
        """
        t1_start = time.time()
        if torch.cuda.is_available():
            x = torch.from_numpy(data_lin.x).cuda()
            y = torch.from_numpy(data_lin.y).cuda()
            t1_start = time.time()
            for i in range(nfits):
                p, q, j = fitter_torch.curve_fit(data.testfunc_lin_torch, x, y, [
                    torch.ones(2, requires_grad=True, dtype=torch.float64, device="cuda:0")])
            t1_stop = time.time()
            comp_time_lin.append(t1_stop - t1_start)
        """
        linlist.append(comp_time_lin)
    name = "lin"
    df = create_benchmark_df(name,optimizers,params_true,params,covs,npoints,weights_index)
    return linlist,df


def benchmark_sin(pointlist):

    params = []
    covs = []
    params_true = []
    optimizers = []
    npoints = []
    weights_index = []

    for idx, el in enumerate(pointlist):

        comp_time_sin = []
        #weights = np.random.rand(nfits,el)*2
        #weights = np.random.poisson(lam=1., size=(nfits, el))
        weights = bootstrap_weights(nfits,el)+eps
        weights[0] = np.ones(el)
        sin_fitter = bfgsfitter(data.testfunc_sin)
        data_sin.setxy(el,sigma0)

        # bfgsfitter
        t1_start = time.time()

        for i in range(nfits):
            p, q = sin_fitter.curve_fit(data_sin.x, data_sin.y)

        t1_stop = time.time()
        comp_time_sin.append(t1_stop - t1_start)
        # print(p)
        # print(q)
        weights_index.append(0)
        params.append(p.numpy())
        covs.append(np.diag(q.numpy()))
        params_true.append(data_sin.params)
        optimizers.append("Tensorflow_BFGS")
        npoints.append(el)

        # second fit after initializiation
        t1_start = time.time()
        for i in range(nfits):
            p, q = sin_fitter.curve_fit(data_sin.x, data_sin.y)

        t1_stop = time.time()
        comp_time_sin.append(t1_stop - t1_start)

        # scipy
        t1_start = time.time()

        for i in range(nfits):
            p, q = scipy.optimize.curve_fit(data.testfunc_sin_np, data_sin.x, data_sin.y,sigma=sigma0/weights[i])
            params.append(p)
            covs.append(np.diag(q))
            params_true.append(data_sin.params)
            optimizers.append("Scipy_LM")
            npoints.append(el)
            weights_index.append(i)
            chisq.append(np.sum(((data.testfunc_sin_np(data_sin.x,*p)-data_sin.y)/sigma0)**2))
            chisq_transformed.append(np.sum(weights[i]*((data.testfunc_sin_np(data_sin.x,*p)-data_sin.y)/sigma0)**2))

        t1_stop = time.time()
        comp_time_sin.append(t1_stop - t1_start)
        # print(p)
        # print(q)


        # pytorch CPU
        t1_start = time.time()
        x = torch.from_numpy(data_sin.x)
        y = torch.from_numpy(data_sin.y)
        for i in range(nfits):
            p, q = fitter_torch.curve_fit(data.testfunc_sin_torch, x, y,
                                             [torch.ones(3, requires_grad=True, dtype=torch.float64)],weights=torch.from_numpy(weights[i]/sigma0**2))
            params.append(p[0].detach().numpy())
            covs.append(np.diag(q.numpy()))
            params_true.append(data_sin.params)
            optimizers.append("PyTorch_LBFGS")
            npoints.append(el)
            weights_index.append(i)
            chisq.append(np.sum(((data.testfunc_sin_np(data_sin.x,*p[0].detach().numpy())-data_sin.y)/sigma0)**2))
            chisq_transformed.append(np.sum(weights[i]*((data.testfunc_sin_np(data_sin.x,*p[0].detach().numpy())-data_sin.y)/sigma0)**2))

        t1_stop = time.time()
        print(p)
        print(q)
        print(t1_stop - t1_start)
        comp_time_sin.append(t1_stop - t1_start)
        sinlist.append(comp_time_sin)
    name = "sin"
    df = create_benchmark_df(name,optimizers,params_true,params,covs,npoints,weights_index)
    return sinlist,df


def benchmark_epx(pointlist):

    params = []
    covs = []
    params_true = []
    optimizers = []
    npoints = []
    weights_index = []

    for idx, el in enumerate(pointlist):
        comp_time_exp = []
        #weights = np.random.rand(nfits,el)*2
        #weights = np.random.poisson(lam=1., size=(nfits, el))
        weights = bootstrap_weights(nfits,el)+eps
        weights[0] = np.ones(el)
        exp_fitter = bfgsfitter(data.testfunc_exp)
        data_exp.setxy(el,sigma0)
        # bfgsfitter
        t1_start = time.time()

        for i in range(nfits):
            p, q = exp_fitter.curve_fit(data_exp.x, data_exp.y)

        t1_stop = time.time()
        comp_time_exp.append(t1_stop - t1_start)
        print(p.numpy())
        print(q.numpy())
        weights_index.append(0)
        params.append(p.numpy())
        covs.append(np.diag(q.numpy()))
        params_true.append(data_exp.params)
        optimizers.append("Tensorflow_BFGS")
        npoints.append(el)

        # second fit after initializiation
        t1_start = time.time()

        for i in range(nfits):
            p, q = exp_fitter.curve_fit(data_exp.x, data_exp.y)

        t1_stop = time.time()
        comp_time_exp.append(t1_stop - t1_start)

        # scipy
        t1_start = time.time()

        for i in range(nfits):
            p, q = scipy.optimize.curve_fit(data.testfunc_exp_np, data_exp.x, data_exp.y,sigma=sigma0/weights[i])
            params.append(p)
            covs.append(np.diag(q))
            params_true.append(data_exp.params)
            optimizers.append("Scipy_LM")
            npoints.append(el)
            weights_index.append(i)
            chisq.append(np.sum(((data.testfunc_exp_np(data_exp.x,*p)-data_exp.y)/sigma0)**2))
            chisq_transformed.append(np.sum(weights[i]*((data.testfunc_exp_np(data_exp.x,*p)-data_exp.y)/sigma0)**2))
        t1_stop = time.time()
        comp_time_exp.append(t1_stop - t1_start)
        print(p)
        print(q)


        # pytorch CPU
        t1_start = time.time()
        x = torch.from_numpy(data_exp.x)
        y = torch.from_numpy(data_exp.y)
        for i in range(nfits):
            p, q = fitter_torch.curve_fit(data.testfunc_exp_torch, x, y,
                                             [torch.ones(2, requires_grad=True, dtype=torch.float64)],weights=torch.from_numpy(weights[i]/sigma0**2))
            params.append(p[0].detach().numpy())
            covs.append(np.diag(q.numpy()))
            params_true.append(data_exp.params)
            optimizers.append("PyTorch_LBFGS")
            npoints.append(el)
            weights_index.append(i)
            chisq.append(np.sum(((data.testfunc_lin_np(data_exp.x,*p[0].detach().numpy())-data_exp.y)/sigma0)**2))
            chisq_transformed.append(np.sum(weights[i]*((data.testfunc_exp_np(data_exp.x,*p[0].detach().numpy())-data_exp.y)/sigma0)**2))

        t1_stop = time.time()
        comp_time_exp.append(t1_stop - t1_start)
        print(p[0].detach().numpy())
        print(q.numpy())
        print(t1_stop - t1_start)


        explist.append(comp_time_exp)

    name = "exp"
    df = create_benchmark_df(name,optimizers,params_true,params,covs,npoints,weights_index)

    return explist,df


linlist,lindf = benchmark_linear(pointlist)
sinlist,sindf = benchmark_sin(pointlist)
explist,expdf = benchmark_epx(pointlist)

#
#plotlists = [explist]
#funcnames = ["exp"]

plotlists = [linlist, sinlist, explist]
funcnames = ["linear", "sinus", "exponential"]

for idx, plotlist in enumerate(plotlists):
    print(funcnames[idx])
    ylist = np.array(plotlist)
    plt.loglog(pointlist, ylist[:, 1])
    plt.plot(pointlist, ylist[:, 2])
    plt.plot(pointlist, ylist[:, 3])
    plt.legend(["tensorflow", "scipy", "pytorch CPU"])
    plt.xlabel("number of points")
    plt.ylabel("time [s]")
    plt.savefig(funcnames[idx] + ".pdf")
    plt.show()

lindf.to_csv("benchmark_linear.csv")
sindf.to_csv("benchmark_sin.csv")
expdf.to_csv("benchmark_exp.csv")

lindf.to_pickle("benchmark_linear.pkl")
sindf.to_pickle("benchmark_sin.pkl")
expdf.to_pickle("benchmark_exp.pkl")
