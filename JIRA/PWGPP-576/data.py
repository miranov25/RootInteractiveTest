import numpy as np
import pandas as pd
import tensorflow as tf
import inspect


def testfunc_lin(x, a, b):
    return a*tf.cast(x,tf.float32)+b

def testfunc_lin_np(x, a, b):
    return a*x+b
    
def testfunc_sin(x, a, b, c):
    return a*tf.sin(b*tf.cast(x,tf.float32))+c

def testfunc_sin_np(x, a, b, c):
    return a*np.sin(b*x)+c

def testfunc_exp(x, a, b):
    return tf.exp(a*tf.cast(x,tf.float32))+b

def testfunc_exp_np(x, a, b):
    return np.exp(a*x)+b
    
class testdata:
    
    def __init__(self):
        self.func = None
        self.x = None
        self.y = None
        self.num_params = None
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
        
    def setxy(self, n):
        self.x = np.array(np.linspace(0,2*np.pi,n))
        y_vals = []
        param_list = []
        for i in range(self.num_params):
            param_list.append(np.random.uniform())
        for el in self.x:
            y_vals.append(np.random.normal(self.func(el, *param_list),0.1))
        y_vals = tf.stack(y_vals).numpy()
        self.y = y_vals
        
    
        
    