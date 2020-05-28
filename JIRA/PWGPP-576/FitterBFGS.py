import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.utils import resample
import numpy as np
import inspect
import pandas as pd

def mse(y_pred, y_true,weights=1):
        """
        standard loss: mean squared error

        @param y_pred: predicted value
        @param y_true: target value
        @return: mse
        """
        return tf.reduce_mean(weights*(y_pred - y_true )**2)


class bfgsfitter:

    def __init__(self, fitfunc, **kwargs):
        
        """
        @param fitfunc: fit function f(x, a, b ,..), x can be multidimensional (must be list), other parameters are fit parameters, f can also be multidimensional
        @param kwargs: options
        @return: values of fitted parameters, the loss and list with parameters
        """
        
        options = {
                "weights": {},
                "initial_parameters": [],
                "loss": mse,
                "tolerance": 1e-5
            }
        options.update(kwargs)

        if type(options["weights"]) != dict:
                w = options["weights"]
                options["weights"] = {}
                options["weights"]["weights"] = w

        # create parameter list
        NUM_PARAMS = len(inspect.getfullargspec(fitfunc)[0])-1
        #self.paramlist = paramlist
        self.fitfunc = fitfunc
        self.options = options
        self.NUM_PARAMS = NUM_PARAMS
        self.y_pred = None
        self.y_true = None
        self.x      = None
        self.tolerance = options["tolerance"]
        
    def loss(self,paramlist):
        self.y_pred = self.fitfunc(self.x,*tf.unstack(paramlist))
        if "weights" in self.options["weights"]:
            return self.options["loss"](self.y_pred,self.y_true,self.options["weights"]["weights"])
        return self.options["loss"](self.y_pred,self.y_true)
    
    def quadratic_loss_and_gradient(self,x):
        return tfp.math.value_and_gradient(self.loss,x)
    
    @tf.function(experimental_compile=True)
    def optim(self):
        return tfp.optimizer.bfgs_minimize(self.quadratic_loss_and_gradient, initial_position=self.paramlist, tolerance=self.tolerance)
            
    
#    @tf.function(experimental_compile=True)
    def curve_fit(self, x, y, **kwargs):
        """
        curve fitting with error calculation via covariance matrix

        @param x: input parameters
        @param y: output parameters
        @return: values of fitted parameters, covariance matrix
        """
        # covariance matrix error calculation via hessian matrix

        
        options = {
                "initial_parameters": [],
            }
        options.update(kwargs)
        self.x = tf.transpose(x)  # right shape..
        self.y_true=y
        self.paramlist = tf.ones([self.NUM_PARAMS])
        if len(options["initial_parameters"])!=0:
            self.paramlist = tf.stack(options["initial_parameters"])

        optim_results = self.optim()
    
        params = optim_results[4]
        cov = (4*optim_results[7]/x.shape[0])*self.loss(params)
        
        return params, cov
    
    def curve_fit_array(self, x, y, **kwargs):
        """
        curve fit of an array of inputs and an array of outputs. Fits are derived in parallel.

        @param x: input parameters as list with length as number of fits
        @param y: output parameters as list with length as number of fits
        @return: values of fitted parameters, covariance matrix as list
        """
        
        elems = (np.array(x).astype('float32'), np.array(y).astype('float32'))
        
        @tf.function
        def mypara(e):
            out=tf.map_fn(lambda x: self.curve_fit(x[0],x[1]), e)
            return out
        
        out=mypara(elems)
        return out


    def curve_fit_BS(self,x, y, init_params,sigma0=1,weights=None,nbootstrap=5,fitter_options={},fitter_name='Tensorflow_BFGS',**kwargs):
        """
        curve fitting with error calculation via bootstrapping

        @param x: input parameters
        @param y: output parameters
        @param fitfunc: fit function f(x, a, b ,..), x can be multidimensional (must be list), other parameters are fitparameters, f can also be multidimensional
        @param kwargs: options
        @return: values of fitted parameters, errors from bootstrapping

        """

        npoints = y.shape[0]
        if weights is None:
            weights = bootstrap_weights(nbootstrap,npoints)
        # error calculation via Bootstrapping
        paramsBS = []
        errorsBS = []
        weights_idx=[]
        chisq=[]
        chisq_transformed=[]
        for i in range(nbootstrap):
            #sample = resample(np.array([x, y]).transpose()).transpose()
            p,q = self.curve_fit(x, y, weights=weights[i]/sigma0**2,**kwargs)
            paramsBS.append(p.numpy())
            errorsBS.append(np.sqrt(np.diag(q.numpy())))
            weights_idx.append(i)
            chisq.append(tf.reduce_sum(((self.y_pred - self.y_true )/sigma0)**2))
            if "weights" in self.options["weights"]:
                chisq_transformed.append(self.options["loss"](self.y_pred,self.y_true,self.options["weights"]["weights"]).numpy())
            else:
                chisq_transformed.append(chisq[-1])
        df = create_benchmark_df(fitter_name,paramsBS,errorsBS,npoints,weights_idx,chisq,chisq_transformed) 
        return df,weights

def create_benchmark_df(optimizers,params,covs,npoints,idx,chisq,chisq_transformed):
    params = np.stack(params)
    covs = np.stack(covs)
    d = {'optimizers':optimizers,'number_points':npoints,'weights_idx':idx,'chisq':chisq,'chisq_transformed':chisq_transformed}
    d.update({str.format("params_{}",i):params[:,i] for i in range(params.shape[1])})
    d.update({str.format("errors_{}",i):covs[:,i] for i in range(covs.shape[1])})
    df = pd.DataFrame(d)
    return df

def bootstrap_weights(nfits,npoints):
    return np.stack([np.bincount(np.random.randint(0,npoints,[npoints]),minlength=npoints) for i in range(nfits)])