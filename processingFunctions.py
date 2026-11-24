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


def loadFile(fileString):
    in_file = str(fileString + '/outputFile.h5')
    with h5py.File(in_file, 'r') as h5f:
        # Read each dataset back into memory
        counts = h5f['counts'][:]
        det_res = h5f['det_res'][:]
        exp = h5f['exp'][()]
        cin_min = h5f['cin_min'][:]
        cin_max = h5f['cin_max'][:]
        cout_min = h5f['cout_min'][:]
        cout_max = h5f['cout_max'][:]
        flux = h5f['flux'][:]
        effA = h5f['effA'][:]

    #We store our dec and ra separately, and unpack them here
    coordsFile = (fileString + '/declensionInfo.txt')
    with open(coordsFile, 'r') as textFile:
        lineIndex = 0
        for line in textFile:
            if lineIndex == 0:
                raDegrees = float(line) 
                lineIndex += 1
                
            elif lineIndex == 1:
                decDegrees = float(line)
                lineIndex += 1
                
            else:
                break

        #Now we convert ra and dec into their values in galactic coordinates
        celestialCoords = SkyCoord(ra=raDegrees * u.degree, dec=decDegrees * u.degree, frame="icrs")
        galacticCoords = celestialCoords.galactic
        lVal = galacticCoords.l.degree
        bVal = galacticCoords.b.degree
    
    E_mean = np.mean([cout_min, cout_max], axis = 0)
    dE = cout_max - cout_min
        
    return counts, det_res, exp, cin_min, cin_max, E_mean, dE, flux, cout_max, cout_min, effA, lVal, bVal

#Our background model
def mod_power_law(ens, x, Exp, DE, eVal):
    logA, B = x
    return (np.exp(logA)*(ens/eVal)**B) * (Exp * DE)

#Other functions that make our graphs run and such
def return_sin_theta_lim(E_line,flux,D_factor):
    DMmass = 2.*E_line
    res = (4.*np.pi*DMmass/D_factor)/1.361e-22*(1/DMmass)**5*flux
    return res

def powerLaw(E, a, b):
    return a*(E**b)

#A class that runs all of the Poisson functions
class poisson:
    """ A set offunctions for calculation the chisq associated with different hypotheses
    """
    def __init__(self,ens,dat,null_mod,sig_template, Exp, DE, centerE):
        self._ens = ens
        self._dat = dat
        self._null_mod = null_mod
        self._sig_template = sig_template
        self._A_sig = 0.0
        self._Exp = Exp
        self._DE = DE
        self.centerE = centerE

    #Basic poisson function. Tests the hypothesis.
    def poisson(self,x): 
        null_mod = self._null_mod(self._ens,x[1:], self._Exp, self._DE, self.centerE)
        sig_mod = self._sig_template*x[0] 
        
        k = self._dat 
        λ = null_mod + sig_mod 

        if np.any(λ <= 0):
            return 1e10 #np.inf
        
        logPoisson = k * np.log(λ) - λ - gammaln(k + 1) 
        
        return -2 * np.sum(logPoisson)

    
    #Similar to poisson, but with no sig_mod; this test only evaluates the null case
    def poissonNull(self,x): 
        null_mod = self._null_mod(self._ens,x, self._Exp, self._DE, self.centerE)
        
        k = self._dat 
        λ = null_mod 
        
        if np.any(λ <= 0):
            return 1e10 #np.inf
            
        logPoisson = k * np.log(λ) - λ - gammaln(k + 1) 
        
        return -2 * np.sum(logPoisson)

    
    #Same as poisson, but with adjusted definitions for the null_mod and sig_mod variables
    def poissonFixedSignal(self,x):
        null_mod = self._null_mod(self._ens,x, self._Exp, self._DE, self.centerE) #,self.centerE)
        sig_mod = self._sig_template*self._A_sig

        k = self._dat 
        λ = null_mod + sig_mod

        if np.any(λ <= 0):
            return 1e10 #np.inf
        
        logPoisson = k * np.log(λ) - λ - gammaln(k + 1) 

        return -2 * np.sum(logPoisson)
    
    def fix_signal_strength(self,A_sig):
        self._A_sig = A_sig

#Functions for processing doppler-broadened signal, per Dr. Kumar
def rho_DM(r):
    r_s = 19.1
    rho_s = 6.6e6
    return rho_s/(r/r_s)/(1+r/r_s)**2

def r(s, l, b):
    r_sun = 8.23
    return sqrt(r_sun**2 + s**2 - 2 * cos(b) * cos(l) * r_sun * s)

def v(E, m_chi):
    c = 3e5
    return (2 * E / m_chi - 1) * c

def f1D(E, m_chi):
    v0 = 220
    return np.exp(-v(E, m_chi)**2/v0**2) / sqrt(pi) / v0

def f_gal(E, m_chi):
    c = 3e5
    return 4 * c / m_chi * f1D(E, m_chi) * np.heaviside(E - m_chi/2, 0)

def f(E, m_chi, l, b):
    nhat = np.array([cos(b) * cos(l), cos(b) * sin(l), sin(b)])
    vsun = np.array([11, 12+220, 7]) # solar velocity in galactic coordinate
    c = 3e5
    return f_gal(E * (1 - np.sum(nhat * vsun)/c), m_chi)

def D_fac(l, b): # in keV/cm^2
    M_sun_in_keV = 1.116e63
    kpc_in_cm = 3.0857e21
    return quad(lambda s: rho_DM(r(s, l, b)), 0, 2000)[0] * M_sun_in_keV / kpc_in_cm**2

def flux(E, m_chi, l, b, tau_chi): # in 1/keV/cm^2/s/sr, tau_chi is in s and m_chi in keV
    return D_fac(l, b) * f(E, m_chi, l, b) / 4 / pi / m_chi / tau_chi

def N_sig(E_in, dE_in, R, m_chi, l, b, tau_chi, Exp):
    phi = flux(E_in, m_chi, l, b, tau_chi) * dE_in

    return Exp * R @ phi
    #return Exp * R[bin_i, :] @ phi

#This actually runs our pipeline
def runPipeline(counts, det_res, R, exp, cin_min, cin_max, E_mean, dE, lineVal, window, tauChi, lval, bval):
    print(f"Running for line value {lineVal}, ALP mass {2*lineVal} keV, ALP half-life {tauChi} seconds")

    binIndex = np.argmin(np.abs(E_mean - lineVal))
    
    Input_Bkg = powerLaw(E_mean, 0.004, 0.5) # cts/cm^2/s/keV
    
    Input_Sig = N_sig(E_mean, dE, R, 2*lineVal, lval, bval, tauChi, exp)
    Input_Model = Input_Bkg + Input_Sig


    OutputBackground = np.matmul(det_res, Input_Bkg)
    OutputSignal = Input_Sig #det_res is already accounted for, so no need to process

    whs_reduced = np.where((E_mean >= lineVal-window) & (E_mean <= lineVal+window))[0]
    E_mean_reduced = E_mean[whs_reduced]
    counts_reduced = counts[whs_reduced]
    OutputSignal_reduced = OutputSignal[whs_reduced]
    dE_reduced = dE[whs_reduced]
    
    #Best fit with just background    
    poisson_instance = poisson(E_mean_reduced,counts_reduced,mod_power_law,(OutputSignal_reduced), exp, dE_reduced, lineVal)
    mn_null = minimize(poisson_instance.poissonNull,np.array([np.log(0.3),0.01]),method='L-BFGS-B',bounds=[(-5, 10), (-5, 5)])

    backgroundParams = mn_null.x

    #Best fit with signal and background
    #mn = minimize(poisson_instance.poisson,np.array([(1.e-2),mn_null.x[0],mn_null.x[1]]),method='L-BFGS-B')
    mn = minimize(
        poisson_instance.poisson,
        np.array([1.e-2, mn_null.x[0], mn_null.x[1]]),
        method='L-BFGS-B',
        bounds=[(0, None),(-5, 10),(-5, 5)],
        options={'ftol': 1e-12,'gtol': 1e-8,'maxiter': 5000,'maxls': 100}
    )
    
    signalParams = mn.x

    aSigBase = np.logspace(np.log10(1e-4), np.log10(3), 200)
    A_sig_array = np.concatenate((np.array([0.0]), aSigBase))
    
    bfList = []

    poisson_sig_array = np.zeros(len(A_sig_array))
    
    # Our starting value based on a detected
    bf = mn_null.x.copy() # Start with the null fit
    #bf = mn.x[1:].copy() # Start with the best overall fit
    
    profile_success = np.zeros(len(A_sig_array), dtype=bool)

    
    for i in range(len(A_sig_array)):
        poisson_instance.fix_signal_strength(A_sig_array[i])
        
        mn_profile = minimize(
            poisson_instance.poissonFixedSignal,
            bf,
            #mn_null.x.copy(), #Trying this out to see if we avoid failures
            method='L-BFGS-B',
            bounds=[(-5, 10), (-5, 5)],
            options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 1000, 'maxls': 50}
        )

        #Adding some defensive programming
        if np.isfinite(mn_profile.fun) and np.all(np.isfinite(mn_profile.x)):
            poisson_sig_array[i] = mn_profile.fun
            bf = mn_profile.x.copy()
            profile_success[i] = True

        else:
            print(
                f"PROFILE FAILED at A={A_sig_array[i]:.6g}: "
                f"success={mn_profile.success}, "
                f"status={mn_profile.status}, "
                f"message={mn_profile.message}, "
                f"fun={mn_profile.fun}, "
                f"x={mn_profile.x}"
            )
            poisson_sig_array[i] = np.inf
        
        bfList.append(bf.copy())

    
    bestFit = np.min(poisson_sig_array)
    bestFitIndex = np.argmin(poisson_sig_array)
    testStat = poisson_sig_array-bestFit

    bestProfileSignal = [A_sig_array[bestFitIndex], bfList[bestFitIndex][0], bfList[bestFitIndex][1]]

    #Get our test statistic for comparing with other line values
    resultTestStat = mn_null.fun - bestFit

    print(f"FOR ENERGY {lineVal}, Test Stat: {resultTestStat}")

    #### NEW #####
    best_idx = np.argmin(testStat)

    upper_teststat = testStat[best_idx:]
    upper_amp = A_sig_array[best_idx:]
    
    crossing = np.flatnonzero(upper_teststat >= 2.71)
    
    if len(crossing) == 0:
        limit_signal_strength = np.nan
        print("No upper-limit crossing found")
    
    elif crossing[0] == 0:
        limit_signal_strength = upper_amp[0]
    
    else:
        i = crossing[0]
    
        limit_signal_strength = np.interp(
            2.71,
            upper_teststat[i-1:i+1],
            upper_amp[i-1:i+1]
        )

    ReducedVals = [whs_reduced,E_mean_reduced,counts_reduced, OutputBackground[whs_reduced], OutputSignal[whs_reduced], dE_reduced]

    #### DIAGNOSTICS ######
    print("TS improvement:", resultTestStat)
    print("Best-fit amplitude:", A_sig_array[bestFitIndex])
    print("Amplitude scan max:", A_sig_array[-1])
    print("Profile min index:", bestFitIndex)
    #### DIAGNOSTICS ######
    
    return backgroundParams, signalParams, bestProfileSignal, resultTestStat, bestFit, bfList, A_sig_array, testStat, limit_signal_strength, ReducedVals
