import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.utils import resample
import numpy as np
import inspect


def mse(y_pred, y_true):
        """
        standard loss: mean squared error

        @param y_pred: predicted value
        @param y_true: target value
        @return: mse
        """
        return tf.reduce_mean((y_pred - y_true )**2)


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
        return self.options["loss"](self.y_pred,self.y_true)
    
    def quadratic_loss_and_gradient(self,x):
        return tfp.math.value_and_gradient(self.loss,x)
    
    @tf.function(experimental_compile=True)
    def optim(self):
        return tfp.optimizer.bfgs_minimize(self.quadratic_loss_and_gradient, initial_position=self.paramlist, tolerance=self.tolerance)
            
    
    @tf.function(experimental_compile=True)
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
        cov = (4*optim_results[7]/len(x))*self.loss(params)
        
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


    def curve_fit_BS(x, y, **kwargs):
        """
        curve fitting with error calculation via bootstrapping

        @param x: input parameters
        @param y: output parameters
        @param fitfunc: fit function f(x, a, b ,..), x can be multidimensional (must be list), other parameters are fitparameters, f can also be multidimensional
        @param kwargs: options
        @return: values of fitted parameters, errors from bootstrapping

        """
        options = {
            "BS_samples": 5
        }
        options.update(kwargs)

        # error calculation via Bootstrapping
        BS_samples = options["BS_samples"]
        paramsBS = []
        for i in range(BS_samples):
            sample = resample(np.array([x, y]).transpose()).transpose()
            numpylist,_ = curve_fit(tf.stack(sample[0]), sample[1], **kwargs)
            paramsBS.append(numpylist.numpy())
        paramsBS = np.array(paramsBS)
        return paramsBS.mean(axis=0), paramsBS.std(axis=0)
