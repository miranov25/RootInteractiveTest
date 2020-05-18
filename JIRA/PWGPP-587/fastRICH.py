import numpy as np;
import numpy as np
import scipy.optimize
from iminuit import Minuit
import time
#import matplotlib.pyplot as plt
# import torch

def angleRICH(detetoctorParam, p, mass, Z, phi, eta):
    """
    :param detetoctorParam:
    :param p:                 particle momentum
    :param mass:              particle mass
    :param Z:                 charge
    :param phi:               inclination angle in phi
    :param eta:               inclination angle eta
    :return:                  angle
    """
    return

def angularResolutionRICH(detectorParam, p, mass, phi,eta):
    """
    :param detectorParam:
        1.) aerogelWidthArray,aerogelRefIndexArray  - multi-layer aerogel with increasing refractive index vector:  [[widthi, ni]]
            * width in cm
            * ref index
        2.) expansion gap (in cm)
        3.) pixel size
        4.) single photon time resolution
        5.) amount of Cherenkov photons/cm (derived from )
        6.) layer radius
    :param p:
    :param mass:
    :param phi:
    :param eta:
    :return: xxx
    """
    #length0=sum()
    #length=length0*sqrt(1+tan(phi)*tan(phi)+ tan(eta)*tan(eta))

    #nCherenkov=length
    return 1

def angleRICHTrack(detectorParam, trackParam):
    return
def angularResolutionRICHtrack(detectorParam, trackParam):
    return

print(angularResolutionRICH(0,0,0,0,0) )