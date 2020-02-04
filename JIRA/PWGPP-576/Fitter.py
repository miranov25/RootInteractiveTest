import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.utils import resample
import numpy as np


def mse(y_pred, y_true):
    """
    standard loss: mean squared error

    @param y_pred: predicted value
    @param y_true: target value
    @return: mse
    """
    return tf.reduce_mean((y_pred - y_true )**2)


def get_hessian(f, x):
    """
    Function calculated hessian/2.
    Helper functions to be used for covariance calculation for curve fit.

    @param f: function for which hessian should be calculated
    @param x: parameters from which to calculate the derivatives
    @return: hessian/2
    """
    # Function to obtain hessian matrix. Direct calculation via tf.hessian does not work in eager execution.
    with tf.GradientTape(persistent=True) as hess_tape:
        with tf.GradientTape() as grad_tape:
            y = f(x)
        grad = grad_tape.gradient(y, x)
        grad_grads = [hess_tape.gradient(g, x) for g in grad]
    hess_rows = [tf.stack(gg)[tf.newaxis, ...] for gg in grad_grads]
    hessian = tf.concat(hess_rows, axis=0)
    return hessian / 2.


def calc_cov(f, x, loss):
    """
    helper functions to be used for covariance calculation for curve fit
    calculation of covariance matrix

    @param f: function for which hessian should be calculated
    @param x: parameters from which to calculate the derivatives
    @param loss: loss value
    @return: covariance matrix
    """


    return tf.linalg.inv(get_hessian(f, x)) * tf.cast(loss, tf.float32)


def curve_fit_raw(x, y, fitfunc, **kwargs):
    """
    performs only fit without errors

    @param x: input parameters
    @param y: output parameters
    @param fitfunc: fit function f(x, a, b ,..), x can be multidimensional (must be list), other parameters are fitparameters, f can also be multidimensional
    @param kwargs: options
    @return: values of fitted parameters, the loss and list with parameters
    """
    options = {
        "weights": {},
        "initial_parameters": [],
        "num_steps": 1000,
        "learning_rate": 0.01,
        "loss": mse
    }
    options.update(kwargs)

    if type(options["weights"]) != dict:
        w = options["weights"]
        options["weights"] = {}
        options["weights"]["weights"] = w

        
    # raw fitting without errors
    x = np.transpose(x)  # right shape..

    # create parameter list
    NUM_PARAMS = fitfunc.__code__.co_argcount - 1
    #print("NUM_PARAMS:", NUM_PARAMS)
    paramlist = options["initial_parameters"]
    if(len(options["initial_parameters"])==0):
        for el in range(NUM_PARAMS):
            paramlist.append(tf.Variable(1.))

    # calculate loss function
    # weightdict = {"weights": options["weights"]}
    loss_fn = lambda: options["loss"](fitfunc(x, *paramlist), tf.transpose(y), **options["weights"])

    # minimize loss
    losses = tfp.math.minimize(loss_fn, num_steps=options["num_steps"], optimizer=tf.optimizers.Adam(learning_rate=options["learning_rate"]),
                               trainable_variables=paramlist)

    # write parameters to numpy array
    numpylist = []
    for el in paramlist:
        numpylist.append(el.numpy())

    return numpylist, losses, paramlist


def curve_fit(x, y, fitfunc, **kwargs):
    """
    curve fitting with error calculation via covariance matrix

    @param x: input parameters
    @param y: output parameters
    @param fitfunc: fit function f(x, a, b ,..), x can be multidimensional (must be list), other parameters are fitparameters, f can also be multidimensional
    @param kwargs: options
    @return: values of fitted parameters, covariance matrix
    """
    # covariance matrix error calculation via hessian matrix

    options = {
        "weights": {},
        "initial_parameters": [],
        "loss": mse
    }
    options.update(kwargs)


    numpylist, losses, paramlist = curve_fit_raw(x, y, fitfunc, **kwargs)
    # calculate covariance matrix

    x = np.transpose(x)  # right shape..
    y = np.transpose(y)

    if type(options["weights"]) != dict:
        w = options["weights"]
        options["weights"] = {}
        options["weights"]["weights"] = w
        
    def covfunc(paramlistcov):
        return options["loss"](fitfunc(x, *paramlist), y, **options["weights"])

    COV = calc_cov(covfunc, paramlist, losses[-1])
    COV = COV.numpy()
    return numpylist, COV


def curve_fit_BS(x, y, fitfunc, **kwargs):
    """
    curve fitting with error calculation via bootstrapping

    @param x: input parameters
    @param y: output parameters
    @param fitfunc: fit function f(x, a, b ,..), x can be multidimensional (must be list), other parameters are fitparameters, f can also be multidimensional
    @param kwargs: options
    @return: values of fitted parameters, errors from bootstrapping

    """
    options = {
        "weights": {},
        "BS_samples": 5
    }
    options.update(kwargs)

    # error calculation via Bootstrapping
    BS_samples = options["BS_samples"]
    paramsBS = []
    for i in range(BS_samples):
        sample = resample(np.array([x, y]).transpose()).transpose()
        numpylist, _, _ = curve_fit_raw(tf.stack(sample[0]), sample[1], fitfunc, **kwargs)
        paramsBS.append(numpylist)
    paramsBS = np.array(paramsBS)
    return paramsBS.mean(axis=0), paramsBS.std(axis=0)
