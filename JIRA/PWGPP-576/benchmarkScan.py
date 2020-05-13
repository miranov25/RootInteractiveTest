# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:25:28 2020

@author: mkroe. majoi
"""
# TODO - parameterize optimizer_options and fit_options (starting from pytorch)

import numpy as np
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


def benchmark_linear(pointList):
    
    params = []
    covs = []
    params_true = []
    optimizers = []
    npoints = []

    for idx, el in enumerate(pointList):
        comp_time_lin = []
        lin_fitter = bfgsfitter(data.testfunc_lin)
        data_lin.setxy(el)
        lin_parameter_list_org.append(data_lin.params)

        # bfgsfitter
        t1_start = time.time()
        for i in range(nfits):
            p, q = lin_fitter.curve_fit(data_lin.x, data_lin.y)
            if (i == 0):
                lin_parameter_list.append(p)
                lin_parameter_list_sigma.append(q)
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(p)
        print(q)
        print(t1_stop - t1_start)
        params.append(p.numpy())
        covs.append(np.diag(q.numpy()))
        params_true.append(data_exp.params)
        optimizers.append("Tensorflow_BFGS")
        npoints.append(el)

        # second fit after initializiation
        t1_start = time.time()
        for i in range(nfits):
            p, q = lin_fitter.curve_fit(data_lin.x, data_lin.y)
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(t1_stop - t1_start)

        # scipy
        t1_start = time.time()
        for i in range(nfits):
            p, q = scipy.optimize.curve_fit(data.testfunc_lin_np, data_lin.x, data_lin.y)
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(p)
        print(q)
        print(t1_stop - t1_start)
        params.append(p)
        covs.append(np.diag(q))
        params_true.append(data_exp.params)
        optimizers.append("Scipy_LM")
        npoints.append(el)
        
        # pytorch CPU
        t1_start = time.time()
        x = torch.from_numpy(data_lin.x)
        y = torch.from_numpy(data_lin.y)
        for i in range(nfits):
            p, q = fitter_torch.curve_fit(data.testfunc_lin_torch, x, y,
                                             [torch.ones(2, requires_grad=True, dtype=torch.float64)])
        t1_stop = time.time()
        comp_time_lin.append(t1_stop - t1_start)
        print(p)
        print(q)
        print(t1_stop - t1_start)
        params.append(p[0].detach().numpy())
        covs.append(np.diag(q.numpy()))
        params_true.append(data_exp.params)
        optimizers.append("PyTorch_LBFGS")
        npoints.append(el)
        
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
    df = pd.DataFrame({'optimizers':optimizers,'params_true':params_true,'params':params,'errors':covs,'number_points':npoints})
    return linlist,df


def benchmark_sin(pointlist):
    
    params = []
    covs = []
    params_true = []
    optimizers = []
    npoints = []
    
    for idx, el in enumerate(pointlist):
        comp_time_sin = []

        sin_fitter = bfgsfitter(data.testfunc_sin)
        data_sin.setxy(el)

        # bfgsfitter
        t1_start = time.time()

        for i in range(nfits):
            p, q = sin_fitter.curve_fit(data_sin.x, data_sin.y)

        t1_stop = time.time()
        comp_time_sin.append(t1_stop - t1_start)
        # print(p)
        # print(q)
        params.append(p.numpy())
        covs.append(np.diag(q.numpy()))
        params_true.append(data_exp.params)
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
            p, q = scipy.optimize.curve_fit(data.testfunc_sin_np, data_sin.x, data_sin.y)

        t1_stop = time.time()
        comp_time_sin.append(t1_stop - t1_start)
        # print(p)
        # print(q)
        params.append(p)
        covs.append(np.diag(q))
        params_true.append(data_exp.params)
        optimizers.append("Scipy_LM")
        npoints.append(el)
        
        # pytorch CPU
        t1_start = time.time()
        x = torch.from_numpy(data_sin.x)
        y = torch.from_numpy(data_sin.y)
        for i in range(nfits):
            p, q = fitter_torch.curve_fit(data.testfunc_sin_torch, x, y,
                                             [torch.ones(3, requires_grad=True, dtype=torch.float64)])
        t1_stop = time.time()
        print(p)
        print(q)
        print(t1_stop - t1_start)
        comp_time_sin.append(t1_stop - t1_start)
        sinlist.append(comp_time_sin)
        params.append(p[0].detach().numpy())
        covs.append(np.diag(q.numpy()))
        params_true.append(data_exp.params)
        optimizers.append("PyTorch_LBFGS")
        npoints.append(el)
    df = pd.DataFrame({'optimizers':optimizers,'params_true':params_true,'params':params,'errors':covs,'number_points':npoints})
    return sinlist,df


def benchmark_epx(pointlist):
    
    params = []
    covs = []
    params_true = []
    optimizers = []
    npoints = []

    for idx, el in enumerate(pointlist):
        comp_time_exp = []

        exp_fitter = bfgsfitter(data.testfunc_exp)
        data_exp.setxy(el)
        # bfgsfitter
        t1_start = time.time()

        for i in range(nfits):
            p, q = exp_fitter.curve_fit(data_exp.x, data_exp.y)

        t1_stop = time.time()
        comp_time_exp.append(t1_stop - t1_start)
        print(p.numpy())
        print(q.numpy())
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
            p, q = scipy.optimize.curve_fit(data.testfunc_exp_np, data_exp.x, data_exp.y)

        t1_stop = time.time()
        comp_time_exp.append(t1_stop - t1_start)
        print(p)
        print(q)
        params.append(p)
        covs.append(np.diag(q))
        params_true.append(data_exp.params)
        optimizers.append("Scipy_LM")
        npoints.append(el)
        
        # pytorch CPU
        t1_start = time.time()
        x = torch.from_numpy(data_exp.x)
        y = torch.from_numpy(data_exp.y)
        for i in range(nfits):
            p, q = fitter_torch.curve_fit(data.testfunc_exp_torch, x, y,
                                             [torch.ones(2, requires_grad=True, dtype=torch.float64)])
        t1_stop = time.time()
        comp_time_exp.append(t1_stop - t1_start)
        print(p[0].detach().numpy())
        print(q.numpy())
        print(t1_stop - t1_start)
        params.append(p[0].detach().numpy())
        covs.append(np.diag(q.numpy()))
        params_true.append(data_exp.params)
        optimizers.append("PyTorch_LBFGS")
        npoints.append(el)
        
        explist.append(comp_time_exp)
    
    df = pd.DataFrame({'optimizers':optimizers,'params_true':params_true,'params':params,'errors':covs,'number_points':npoints})
    
    return explist,df


#linlist = benchmark_linear(pointlist)
#sinlist = benchmark_sin(pointlist)
explist,df = benchmark_epx(pointlist)

#
plotlists = [explist]
funcnames = ["exp"]

#plotlists = [linlist, sinlist, explist]
#funcnames = ["linear", "sinus", "exponential"]

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
