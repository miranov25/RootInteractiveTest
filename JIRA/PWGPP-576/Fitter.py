import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.utils import resample
import numpy as np


def get_hessian(f, x):
    # Function to obtain hessian matrix. Direct calculation via tf.hessian does not work in eager execution.
    with tf.GradientTape(persistent=True) as hess_tape:
        with tf.GradientTape() as grad_tape:
            y = f(x)
        grad = grad_tape.gradient(y, x)
        grad_grads = [hess_tape.gradient(g, x) for g in grad]
    hess_rows = [tf.stack(gg)[tf.newaxis, ...] for gg in grad_grads]
    hessian = tf.concat(hess_rows, axis=0)
    return hessian/2.

def calc_cov(f, x, loss):
    # calculation of covariance matrix
    
    return tf.linalg.inv(get_hessian(f,x))*tf.cast(loss,tf.float32)

def curve_fit_raw(x, y, fitfunc, lossfunc, **kwargs):
    
    
    options = {
        "weights" : {}}
    options.update(kwargs)
        
        
    if type(options["weights"])!=dict:
        w = options["weights"]
        options["weights"] = {}
        options["weights"]["weights"]=w
        
    #raw fitting without errors
    x=np.transpose(x) # right shape..
    
    # create parameter list
    NUM_PARAMS = fitfunc.__code__.co_argcount-1
    print("NUM_PARAMS:",NUM_PARAMS)
    paramlist = []
    for el in range(NUM_PARAMS):
        paramlist.append(tf.Variable(1.))
        
    # calculate loss function
    #weightdict = {"weights": options["weights"]}
    loss_fn = lambda: lossfunc(fitfunc(x, *paramlist), tf.transpose(y), **options["weights"])
    
    # minimize loss
    losses = tfp.math.minimize(loss_fn, num_steps=1000, optimizer=tf.optimizers.Adam(learning_rate=0.01),trainable_variables=paramlist)
    
    # write parameters to numpy array
    numpylist = []
    for el in paramlist:
        numpylist.append(el.numpy())
    
    return numpylist, losses, paramlist

def curve_fit(x, y, fitfunc, lossfunc, **kwargs):
    # covariance matrix error calculation via hessian matrix
    
    options = {
        "weights" : {}}
    options["weights"].update(kwargs)
    
    weightdict = {}
    numpylist, losses, paramlist = curve_fit_raw(x, y, fitfunc, lossfunc, **kwargs)
    # calculate covariance matrix
    
    x=np.transpose(x) # right shape..
    y=np.transpose(y)
    def covfunc(paramlistcov):
        return lossfunc(fitfunc(x, *paramlistcov), y, **options["weights"])
    
    COV = calc_cov(covfunc, paramlist, losses[-1])
    COV = COV.numpy()
    return numpylist, COV

def curve_fit_BS(x, y, fitfunc, lossfunc, **kwargs):
    
    options = {
        "weights" : {}}
    options["weights"].update(kwargs)
    
    # error calculation via Bootstrapping
    BS_samples = 5
    paramsBS = []
    for i in range(BS_samples):
        sample = resample(np.array([x, y]).transpose()).transpose()
        numpylist,_ ,_= curve_fit_raw(tf.stack(sample[0]), sample[1], fitfunc, lossfunc, **kwargs)
        paramsBS.append(numpylist)
    paramsBS = np.array(paramsBS)
    return paramsBS.mean(axis=0), paramsBS.std(axis=0)