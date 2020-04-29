# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:34:01 2020

@author: majoi
"""

import torch
import torch.nn
import torch.optim

def mse(x,y):
    return torch.mean((x-y)**2)

def curve_fit(fitfunc,x,y,params, optimizer_options={}, **kwargs):
    
    options = {
            "atol":1e-3,
            "max_iterations":1000,
            "optimizer":torch.optim.LBFGS,
            "lossfunc":mse
            }
    options.update(kwargs)
    
    optimizer = options["optimizer"](params, **optimizer_options)
    
    atol = options["atol"]
    lossfunc = options["lossfunc"]
    
    for i in range(options["max_iterations"]):
        def closure():
            optimizer.zero_grad()
            y_appx = fitfunc(x,*params)
            loss = lossfunc(y,y_appx)
            loss.backward()
            return loss
        optimizer.step(closure)
    
        optimizer.zero_grad()
        y_appx = fitfunc(x,*params)
        loss = lossfunc(y,y_appx)
        loss.backward()
        ngrad=torch.norm(params[0].grad**2)/params[0].numel()
        if ngrad<atol:
            break
   # print(i)        
    
    
    return params,loss.detach(),i