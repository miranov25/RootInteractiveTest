# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 20:51:39 2020
a[]
@author: mkroe
"""

import numpy as np
import tensorflow as tf
import inspect
import torch

def testfunc_lin(x, a, b):
    return a*tf.cast(x,tf.float32)+b

def testfunc_lin_np(x, a, b):
    return a*x+b

def testfunc_lin_torch(x,a):
    return a[0]*x+a[1]
    
def testfunc_sin(x, a, b, c):
    return a*tf.sin(b*tf.cast(x,tf.float32))+c

def testfunc_sin_np(x, a, b, c):
    return a*np.sin(b*x)+c

def testfunc_sin_torch(x, a):
    return a[0]*torch.sin(a[1]*x)+a[2]

def testfunc_exp(x, a, b):
    return tf.exp(a*x)*b

def testfunc_exp_np(x, a, b):
    return np.exp(a*x)*b

def testfunc_exp_torch(x,a,b):
    return torch.exp(a*x)*b

def testfunc_gaus_np(x,a,b,c):
    return np.exp(-.5*((x-b)**2)/a)*c

def testfunc_gaus_tf(x,a,b,c):
    return tf.exp(-.5*((x-b)**2)/a)*c

def testfunc_gaus_torch(x,a,b,c):
    return torch.exp(-.5*torch.square(x-b)/a)*c

def return_array(array_size, n, func):
    data_array = []
    for i in range(array_size):
        dataa = testdata()
        dataa.setfunclin()
        if (func == "exp"):
            dataa.setfuncexp()
        if (func == "sin"):
            dataa.setfuncsin()
        dataa.setxy(n)
        data_array.append(dataa)
    x_array = []
    y_array = []
    for el in data_array:
        x_array.append(el.x)
        y_array.append(el.y)
    return np.array(x_array), np.array(y_array)
    
class testdata:
    
    def __init__(self,seed=None):
        self.func = None
        self.x = None
        self.y = None
        self.num_params = None
        self.params = None
        #self.setfunclin()
        #self.setxy(n)
        
    def setfunclin(self):
        self.func = testfunc_lin
        self.num_params = len(inspect.getfullargspec(self.func)[0])-1
        
    def setfuncexp(self):
        self.func = testfunc_exp
        self.num_params = len(inspect.getfullargspec(self.func)[0])-1
        
    def setfuncsin(self):
        self.func = testfunc_sin
        self.num_params = len(inspect.getfullargspec(self.func)[0])-1
        
    def setxy(self, n, sigma):
        self.x = np.array(np.linspace(0,2*np.pi,n),dtype=np.float32)
        y_vals = []
        param_list = []
        for i in range(self.num_params):
            param_list.append(np.random.uniform())
        for el in self.x:
            y_vals.append(np.random.normal(self.func(el, *param_list),sigma))
        y_vals = np.stack(y_vals).astype(np.float32)
        self.y = y_vals
        self.params = param_list