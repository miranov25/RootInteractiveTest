# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:45:57 2020

@author: majoi
"""

import numpy as np
import torch
from histogramdd import histogramdd
import time

torch.random.manual_seed(19680801)

sample = torch.arange(0,1,.01)*torch.arange(1,4).reshape([-1,1]) % 1
histnp = np.histogramdd(sample.transpose(0,1))[0]
hist = histogramdd(sample)[0]
print(hist-histnp)

sample = torch.rand(3,100000)
histnp = np.histogramdd(sample.transpose(0,1),6)[0]
hist = histogramdd(sample,6)[0]
print(hist-histnp)

sample = torch.rand(3,100000)
histnp = np.histogramdd(sample.transpose(0,1),[4,5,6])[0]
hist = histogramdd(sample,[4,5,6])[0]
print(hist-histnp)

sample = torch.rand(3,100000)
ranges = torch.empty(2,3)
ranges[0,:] = .2
ranges[1,:] = .8
histnp = np.histogramdd(sample.transpose(0,1),[4,5,6],[[.2,.8],[.2,.8],[.2,.8]])[0]
hist = histogramdd(sample,[4,5,6],ranges)[0]
print(hist-histnp)

sample = torch.rand(3,100000)
bins = [torch.rand([8]).sort()[0],torch.rand([7]).sort()[0],torch.rand([6]).sort()[0]]
histnp = np.histogramdd(sample.transpose(0,1),bins)[0]
hist = histogramdd(sample,bins)[0]
print(hist-histnp)

sample = torch.rand(2,10000000)
csample = sample.cuda()
t0 = time.time()
hist = histogramdd(sample,2)[0]
print('CPU: histogramdd in %0.3fms' % (1000*(time.time()-t0)))
torch.cuda.synchronize()
t0 = time.time()
histc = histogramdd(csample,2)[0]
torch.cuda.synchronize()
print('GPU: histogramdd in %0.3fms' % (1000*(time.time()-t0)))
dif = hist - histc.to('cpu')
print(dif)

sample = torch.zeros(2,10000000)
sample[:,0] = 1
csample = sample.cuda()
t0 = time.time()
hist = histogramdd(sample,2)[0]
print('CPU: histogramdd in %0.3fms' % (1000*(time.time()-t0)))
torch.cuda.synchronize()
t0 = time.time()
histc = histogramdd(csample,2)[0]
torch.cuda.synchronize()
print('GPU: histogramdd in %0.3fms' % (1000*(time.time()-t0)))
dif = hist - histc.to('cpu')
print(dif)
