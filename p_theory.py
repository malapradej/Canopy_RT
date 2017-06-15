#!/usr/bin/python
"""
A module that contains the methods required in p-theory or spectral invariant
theory. This module calls methods in the radtran module. Requires the 
quad_dict.dat file to get quadratures and weights for the numerical integration 
by gaussian quadratures. 
The module has been supplemented with our code for the inclusion of fluorescence
in p-theory.
"""

import numpy as np
import pickle
import os
import radtran as rt
import matplotlib.pylab as plt

N = 16 # levels
oct_pnts = N * (N + 2) / 8 # nr pnts in an octant
module_dir = os.path.dirname(__file__)
f = open(os.path.join(module_dir,'quad_dict.dat'))
quad_dict = pickle.load(f)
f.close()
quad = quad_dict[str(N)][:oct_pnts,:]
gauss_wt = quad[:,2] # per octant. 
# see Lewis 1984 p.172 eq(4-40) 
gauss_mu = np.cos(quad[:,0])
gauss_sin = np.sin(quad[:,0])
views = np.array(zip(quad[:,0],quad[:,1]))


def p(LAI, arch='s', Disp='pois', par=None, N=None):
    '''A simple way to calculate recollision probability based on the the canopy
    interceptance and LAI. Based on Stenberg (2007) eq 10. 
    Parameters:
        i0 - canopy interceptance
        LAI - leaf area index
    Returns:
        canopy recollision probability.
    '''
    i0 = interceptance(LAI, arch, Disp, par, N)
    p = 1 - i0 / LAI
    if p < 0.:
        p = 0.
    return p

def interceptance(LAI, arch='s', Disp='pois', par=None, N=None):
    '''The canopy interceptance based on the Stenberg (2007) eq 12. Ours makes 
    use of Gaussian quadratures for the hemispherical integration. Based on 
    gap probability being the complement of interceptance.
    Parameters:
        LAI - leaf area index
        arch - archetype (see radtran.py for description)
        Disp - 'pb' positive Binomial, 'nb' negative Binomial, 'pois' Poisson
        par - additional parameters depending on archetype
    Returns:
        Canopy interceptance.
    '''
    Pgap = np.zeros((oct_pnts)) 
    for i, zen in enumerate(views[:,0]):
        Pgap[i] = rt.P0(zen, arch, LAI, Disp, par, N)
        #Pgap[i] = rt.P0(zen, 'l', LAI, 'pois', par=(-0.35, -0.15))

    i0 = sum((1 - Pgap) * gauss_wt)
    return i0

def SIFr_approx(w, p, Fsr):
    '''The approximation of the solar induced fluorescence ratio up to the 4th 
    order of scattering and emission. This is only to show difference 
    between exact Newmann equation and this approximation thus confirming
    the correctness of our method. See notes on 22/7/14 for derivation.
    Input: w - leaf single scattering albedo without SIF, p - canopy recol. prob. 
    Fsr - leaf fluorescence as proportion of intercepted radiation.
    Output: canopy level SIF as fraction of intercepted radiation.
    '''
    Echo = (1-p)*Fsr + (1-p)*(Fsr**2*p + 2*Fsr*p*w) + (1-p)*(Fsr**3*p**2 + \
        3*Fsr**2*p**2*w + 3*Fsr*p**2*w**2) + (1-p)*(4*Fsr**3*p**3*w + \
        6*Fsr**2*p**2*w**2 + 4*Fsr*p**3*w**3 + Fsr**4*p**3)
    return Echo

def Alb_SIFr(w, p, Fsr):
    '''The total canopy albedo including SIF. This is based on the Neumann  
    series we derived in the notes on 28/7/14. The canopy albedo without 
    SIF needs to be subracted to get the canopy level SIF.
    Input: w - leaf single scattering albedo without SIF, p - canopy recol. prob. 
    Fsr - leaf fluorescence as proportion of intercepted radiation.
    Output: canopy level albedo including SIF. 
    '''
    Ws = (1-p) * (w/(1-Fsr*p)/(1-w*p) + Fsr/(1-Fsr*p)/(1-w*p) + \
        Fsr*p**2*w**2/(1-Fsr*p)**2/(1-w*p)**2 + Fsr**2*p**2*w/(1-Fsr*p)**2 \
        /(1-w*p)**2)
    return Ws

def Alb(w, p):
    '''The canopy albedo without SIF. This is the standard formula used 
    in several papers including Knyazikhin, Lewis and others. This is
    subtracted from Alb_SIFr to get the canopy level SIF.
    Input: w - leaf single scattering albedo without SIF, p - canopy recol. prob.
    Output: canopy level albedo without SIF.
    '''
    Wc = w * (1-p) / (1-w*p)
    return Wc

def SIFr(w, p, Fsr):
    '''The canopy level SIF.
    Input: w - leaf single scattering albedo without SIF, p - canopy recol. prob. 
    Fsr - leaf fluorescence as proportion of intercepted radiation.
    Output: canopy level SIF as fraction of intercepted radiation.
    '''
    return Alb_SIFr(w, p, Fsr) - Alb(w, p)

def plot_SIFr(ws, ps, Fsrs):
    '''Plots SIF depending on varying parameters.
    '''
    x = np.size(ws)
    fig = plt.figure()
    fig.subplots_adjust(left=0.1, bottom=0.1, right=.9, top=.9, hspace=0.4)
    for i, w in enumerate(ws):
        ax = plt.subplot(x,1,i+1)
        meshx, meshy = np.meshgrid(ps, Fsrs)
        grid = SIFr(w, meshx, meshy)
        cs = ax.contourf(ps, Fsrs, grid)
        cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
        cbar.ax.set_ylabel('Canopy SIF fraction of PAR')
        ax.set_title('PAR albedo: %.2f ' % w)
        ax.set_xlabel('p (recollision prob.)')
        ax.set_ylabel('Fsr (leaf SIF fraction of PAR)')
    plt.show()


def plot_p():
    '''A test plot of p to compare with the example in Stenberg (2007) fig. 1.
    '''
    LAI = np.concatenate((np.array([0.5, 0.75]), np.arange(1.,11.)), axis=1)
    parr = []
    arch = 's'
    Disp = 'pois'
    for l in LAI:
        rp = p(l, arch, Disp)
        parr.append(rp)
    plt.plot(LAI, parr, 'ok')
    plt.title('Canopy recollision probability',fontsize=18)
    plt.xlabel('Leaf area index', fontsize=16)
    plt.ylabel('p', fontsize=16)
    plt.text(5,0.5,'archetype: %s\ndispersion: %s' %(arch, Disp), fontsize=14)
    plt.show()