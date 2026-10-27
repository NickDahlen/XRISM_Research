#Basic imports
import os,sys
import numpy as np
from astropy.io import fits
import astropy
import h5py
from scipy.special import erf
from scipy.stats import chi2 as chi2_scipy
from scipy.optimize import dual_annealing
from scipy.optimize import minimize
from scipy.integrate import quad
from math import sqrt, cos, exp, pi, sin

from scipy.special import factorial
from scipy.special import gammaln

from astropy.coordinates import SkyCoord
from astropy import units as u
import argparse

#Parsing our arguments
parser = argparse.ArgumentParser(description='Extracts an mChi vs. tauChi plot from our XRISM data')
parser.add_argument('-f','--fileName', help='The 9 digit obsID of the observation we are looking at', required=True)
parser.add_argument('-s','--scale', help='The scale of the observation. Input either 0 for quick, 1 for every 0.05 keV, 2 for every 0.01 keV, 3 for focus on 3.5-3.6 keV', required=True)
args = vars(parser.parse_args())

scale = int(args['scale'])
filePath = str(args['fileName'])
fileName = filePath[-9:]


#Some cosmological/astronimical constants that will let our equations run properly 
rho_s = 6.6e6 # Msun/kpc^3
r_s = 19.1 # kpc
r_sun = 8.23 # kpc
v0 = 220 # in km/s
c = 3e5 # in km/s
M_sun_in_keV = 1.116e63
kpc_in_cm = 3.0857e21

print("Importing custom functions", flush=True)
from processingFunctions import loadFile, mod_power_law, return_sin_theta_lim, powerLaw, poisson, rho_DM, r, v, f1D, f_gal, f, D_fac, flux, N_sig, runPipeline

#First we upload the data
print("Uploading data for file:", fileName, flush=True)
Counts, Det_res, Exp, Cin_min, Cin_max, EMean, dE, Flux, cout_max, cout_min, effA, l, b = loadFile(str(filePath))

#Unpack some useful values from that data
E_in  = 0.5 * (Cin_min + Cin_max)
dE_in = dE

arcsec2_to_sr = 2.350443e-11
roi_size = np.pi * (10 * 60)**2 * arcsec2_to_sr

R = Det_res * roi_size # (n_out, n_in), [cm^2 sr] # response matrix; roi_size is the flatcircle area in steradians, and det_res already contains the effA from arf

lineVals = []

baseVal = 1.65

if scale == 1:
    for x in range(357):
        lineVals.append(baseVal + 0.05 * x)

elif scale == 0:
    for x in range(35):                                                                                                               
        lineVals.append(baseVal + 0.5 * x)

elif scale == 2:
    for x in range(1785):
        lineVals.append(baseVal + 0.01 * x)

elif scale == 3:
    baseVal = 3.5
    for x in range(200):
        lineVals.append(baseVal + 0.005 * x)

else:
    print("Scale all wrong - we'll just default to quick scanning", flush=True)
    for x in range(35):
        lineVals.append(baseVal + 0.5 * x)

bestFitTestStats = []
testStatsDict = {}
mChiVals = [energy * 2 for energy in lineVals]

print("Running the analysis", flush=True)
#Now we run the full analysis
runNo = 0
for lineVal in lineVals:
    print("Run number:", runNo, flush=True)
    runNo += 1
    backgroundParams, signalParams, profileSignal, bestFitTestStat, bestFit, bfList, A_sig_array, testStat, limit_signal_strength, reducedVals = runPipeline(Counts, Det_res, R, Exp, Cin_min, Cin_max, EMean, dE, lineVal, window=0.015, tauChi=1e25, lval=l, bval=b)
    bestFitTestStats.append(bestFitTestStat)
    testStatsDict.update({lineVal:testStat})

print("Analysis complete. Saving results!", flush=True)

if scale == 1:
    out_file = 'profiles/' + fileName + 'Full.h5'

elif scale == 2:
    out_file = 'profiles/' + fileName + 'Deep.h5'

elif scale == 3:
    out_file = 'profiles/' + fileName + '.h5'
    
else:
    out_file = 'profiles/' + fileName + '.h5'
    
h5f = h5py.File(out_file, 'w')
h5f.create_dataset('A_sig_array',data=A_sig_array)
h5f.create_dataset('bestFitTestStats',data=bestFitTestStats)
h5f.create_dataset('mChi',data=mChiVals)
for key, value in testStatsDict.items():
    h5f.create_dataset(str(key), data=value)

h5f.close()

print("Processing successful!", flush=True)
