#!/usr/bin/python

'''This will turn out to be the two-angle discrete-ordinate exact-
kernel finite-difference implimentation of the RT equation as set
out in Myneni 1988d. The script references the radtran.py module
which holds all the functions required for the calculations. It 
also requires the prospect_py.so module created using f2py.
'''

import numpy as np
import scipy as sc
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
import matplotlib.tri as tri
import warnings
import os
from radtran import *
import nose
from progressbar import *
import pdb
import prospect_py # prospect leaf rt module interface


class rt_layers():
  '''The class that will encapsulate the data and methods to be used
  for a single wavelenth of radiative transfer through the canopy. 
  Input: Tol - max tolerance, Iter - max iterations, N - no of 
  discrete ordinates over 0 to pi, K - no of nodes, Lc - total LAI, 
  refl - leaf reflectance, trans - leaf transmittance, refl_s - 
  soil reflectance, F - total flux incident at TOC in W/m2, 
  Beta - fraction of direct solar illumination, sun0_zen - 
  zenith angle of solar illumination, sun0_azi - azimuth angle of 
  solar illumination, arch - leaf angle distribution archetype, 
  par - parameters for leaf angle distribution, ln - number of leaf 
  plate layers, cab - chlorophyl content in g/cm2, car - carotinoid 
  content in g/cm2, cbrown - brown scenesent pigments (units), cw - 
  water content in cm (EWT), cm - dry matter content in g/cm2, lamda - 
  wavelength, CI - clumping index, Ht - canopy height in meters,
  leafd - leaf diameter in meters, n - leaf wax refraction index, 
  k - leaf coeff. of roughness, user_views - array of user views in
  degrees [[zen, azi]] format, sgl - True if only single scattered
  radiation is considered.
  Ouput: a rt_layers class.
  '''

  def __init__(self, Tol = 1.e-6, Iter = 200, K = 20, N = 4,\
      Lc = 2.0, refl_s = 0.0, F = 1., Beta=1.0, sun0_zen = 180.,\
      sun0_azi = 0.0, arch = 'u', par= None, ln = 1.2, cab = 30., \
      car = 10., cbrown = 0., cw = 0.015, cm = 0.009, lamda = 760, \
      refl = 0.2, trans = 0.2, CI = 1.0, Ht=None, leafd=None, \
      n=None, k=None, user_views=None, sgl=False):
    '''The constructor for the rt_layers class.
    See the class documentation for details of inputs.
    '''
    if user_views is not None: 
      if isinstance(user_views, np.ndarray) and \
          np.shape(user_views)[1]==2:
        self.uviews = user_views * np.pi / 180.
      else:
        raise Exception('IncorrectUserViews') 
    self.sgl = sgl
    self.Tol = Tol
    self.Iter = Iter
    self.K = K
    if int(N) % 2 == 1:
      N = int(N)+1
      print 'N rounded up to even number:', str(N)
    self.N = N
    self.Lc = np.float(Lc)
    # choose between PROSPECT or refl trans input
    if np.isnan(refl) or np.isnan(trans):
      self.ln = ln
      self.cab = cab
      self.car = car
      self.cbrown = cbrown
      self.cw = cw
      self.cm = cm
      self.lamda = lamda
      refl, trans = prospect_py.prospect_5b(ln, cab, car, cbrown, cw,\
          cm)[lamda-401]
    else:
      self.ln = np.nan
      self.cab = np.nan
      self.car = np.nan
      self.cbrown = np.nan
      self.cw = np.nan
      self.cm = np.nan
      self.lamda = np.nan
    self.refl = refl
    self.trans = trans
    self.refl_s = refl_s
    self.Beta = Beta
    self.CI = CI
    self.sun0_zen = sun0_zen * np.pi / 180.
    self.sun0_azi = sun0_azi * np.pi / 180.
    self.sun0 = np.array([self.sun0_zen, self.sun0_azi])
    self.mu_s = np.cos(self.sun0_zen)
    self.I0 = Beta * F / -self.mu_s / np.pi # per steridian 
    # eq(11)
    self.F = F
    # it is assumed that total flux at TOC will be provided prior.
    # eq(12)
    self.Id = (1. - Beta) * F / np.pi # per steridian
    self.arch = arch
    self.par = par
    self.albedo = self.refl + self.trans # leaf single scattering albedo
    f = open('quad_dict.dat')
    quad_dict = pickle.load(f)
    f.close()
    quad = quad_dict[str(N)]
    self.gauss_wt = quad[:,2] / 8.# per octant. now wts sum to 1.
    # see Lewis 1984 p.172 eq(4-40) 
    self.gauss_mu = np.cos(quad[:,0])
    self.views = np.array(zip(quad[:,0],quad[:,1]))
    # intervals
    self.dk = self.Lc/self.K
    self.mid_ks = np.arange(self.dk/2.,self.Lc,self.dk)#mid of pairs 
    self.n = N*(N + 2)
    # node arrays and boundary arrays
    self.views_zen = quad[:,0]
    self.views_azi = quad[:,1]
    self.sun_up = self.views[:self.n/2]
    self.sun_down = self.views[self.n/2:]
    # add user views if available
    if user_views is not None: 
      self.un = np.shape(self.uviews)[0]
      self.uviews_up = self.uviews[np.where(self.uviews[:,0] < np.pi/2)]
      self.uviews_dn = self.uviews[np.where(self.uviews[:,0] > np.pi/2)]
      self.nup = np.shape(self.uviews_up)[0]
      self.ndn = np.shape(self.uviews_dn)[0]
      self.ntot = self.n + self.nup + self.ndn
      self.sun_up = np.vstack((self.sun_up, self.uviews_up))
      self.sun_down = np.vstack((self.sun_down, self.uviews_dn))
      self.views = np.vstack((self.sun_up, self.sun_down))
      self.views_zen = self.views[:,0]
      self.views_azi = self.views[:,1] 
      self.mu_v = np.cos(self.views_zen)
      self.index_gauss = np.array([np.arange(0,self.n/2)\
          ,np.arange(self.n/2+self.nup,self.n+self.nup)]).flatten() 
    else:
      self.un = 0.
      self.nup = 0.
      self.ndn = 0.
      self.ntot = self.n
      self.mu_v = self.gauss_mu
      self.index_gauss = np.arange(0,self.n)
    # diffuse sky light refleted by soil eq(15)
    # multiply gauss_wt by 2 due to our upper quadratures summing to 1/2.
    # integral over upper hemisphere should be 1.
    # pi falls away due to per steridian. 
    # I have checked this...
    self.Ird = self.refl_s * -np.sum(np.multiply(\
        np.multiply(self.gauss_mu[self.n/2:],\
        self.I_f(self.sun_down[:self.n/2], self.Lc, self.Id)),\
        self.gauss_wt[self.n/2:] * 2.)) * 2.#* np.pi #pi upper hemisphere
    # direct sun light reflected by soil eq(14)
    self.Ir0 = -self.refl_s * self.mu_s * \
        self.I_f(self.sun0, self.Lc, self.I0)#* np.pi#pi upper hemisphere 
    # discrete ordinate equations eq(29 - 35)
    self.g = G(self.views_zen,self.arch,self.par) * self.CI 
    self.a = (1. + self.g*self.dk/2./self.mu_v)/\
        (1. - self.g*self.dk/2./self.mu_v)
    self.b = (self.g*self.dk/self.mu_v)/\
        (1. - self.g*self.dk/2./self.mu_v)
    self.c = (1. - self.g*self.dk/2./self.mu_v)/\
        (1. + self.g*self.dk/2./self.mu_v)
    self.d = (self.g*self.dk/self.mu_v)/\
        (1. + self.g*self.dk/2./self.mu_v) 
    # build in check for possible negative fluxes.
    # based on Myneni 1988b eq(37).  
    if any((self.g * self.dk / 2. / self.mu_v) > 1.):
      raise Exception('NegativeFluxPossible')
    # G-function cross-sections 
    self.Gx = self.g 
    self.Gs = G(self.sun0_zen,self.arch,self.par) * self.CI 
    self.Inodes = np.zeros((K,3,self.ntot)) # K, upper-mid-lower, N
    self.Jnodes = np.zeros((K,self.ntot))
    self.Q1nodes = self.Jnodes.copy()
    self.Q2nodes = self.Jnodes.copy()
    self.Q3nodes = self.Jnodes.copy()
    self.Q4nodes = self.Jnodes.copy()
    self.Px = np.zeros((self.ntot,self.n)) # P cross-section array
    self.Ps = np.zeros(self.ntot) # P array for solar zenith
    self.I_top_bottom = {}
    # a litte progress bar for long calcs
    widgets = ['Progress: ', Percentage(), ' ', \
      Bar(marker='0',left='[',right=']'), ' ', ETA()] 
    maxval = np.shape(self.views)[0] * (np.shape(self.mid_ks)[0] + \
        np.shape(self.views)[0] + 1) + 1
    pbar = ProgressBar(widgets=widgets, maxval = maxval)
    count = 0
    print 'Setting-up Phase and Q term arrays....'
    pbar.start()
    # hot spot parameter
    self.Ht = Ht
    self.leafd = leafd
    if Ht and leafd:
      self.hspot = self.hotspot()
    else:
      self.hspot = np.ones((self.K, self.ntot))
    # specular refl
    self.spec_n = n
    self.spec_k = k
    # setting up Ps
    for (i,v) in enumerate(self.views):
      count += 1
      pbar.update(count)
      # see if * 2/pi works. it doesn't on its own...
      # tried * 8 to get sum(P2) to 4pi...
      self.Ps[i] = P2(v,self.sun0,self.arch,self.refl,\
        self.trans, self.CI, self.par, self.spec_k, self.spec_n) #* 8 # * 2. / np.pi
    for (i,v1) in enumerate(self.views):
      for (j,v2) in enumerate(self.views[self.index_gauss]):
        count += 1
        pbar.update(count)
        #pdb.set_trace()
        # see if * 2/pi works. it doesn't on its own...  
        self.Px[i,j] = P2(v1, v2 ,self.arch, self.refl, self.trans,\
            self.CI, self.par) #* 8 # * 2 / np. pi
    for (i, k) in enumerate(self.mid_ks):
      for (j, v) in enumerate(self.views):
        count += 1
        pbar.update(count)
        self.Q1nodes[i,j] = self.Q1(j,i)
        #pdb.set_trace()
        self.Q2nodes[i,j] = self.Q2(j,i)
        self.Q3nodes[i,j] = self.Q3(j,i)
        self.Q4nodes[i,j] = self.Q4(j,i)
    self.Bounds = np.zeros((2,self.ntot))
    pbar.finish()
    # alarm to notify of end of instance creation
    #os.system('play --no-show-progress --null --channels 1 \
    #        synth %s sine %f' % ( 0.5, 500)) # ring the bell

  def hotspot(self):
    '''The hotspot factor to be applied to the G-function
    accoring to Marshak 1989 eq(31)''' 
    A0 = G(self.sun0_zen, self.arch, self.par) / np.abs(self.mu_s)
    A = G(self.views_zen[:self.n/2+self.nup], self.arch, self.par) / \
        np.abs(self.mu_v[:self.n/2+self.nup])
    C = np.ones((self.n/2+self.nup))
    for i, v in enumerate(self.views[:self.n/2+self.nup]):
      C[i] = (1./self.mu_s**2 + 1./self.mu_v[i]**2 + \
          2.*dot(self.sun0, v)/np.abs(self.mu_s * \
          self.mu_v[i]))**0.5 
    tau = self.mid_ks 
    kappa = self.leafd / self.Ht # hot spot parameter
    H = self.Lc
    hspot = np.ones((self.K, self.ntot)) 
    for i, t in enumerate(tau):
      hspot[i,:self.n/2+self.nup] = 1. - np.sqrt(A0/A) * \
          np.exp(-C*t/kappa*H)
    return hspot

  def I_down(self, k):
    '''The discrete ordinate downward equation eq(29).
    '''
    n = self.n/2+self.nup
    self.Inodes[k,2,n:] = self.a[n:]*self.Inodes[k,0,n:] - \
        self.b[n:]*(self.Jnodes[k,n:] + self.Q1nodes[k,n:] + \
        self.Q2nodes[k,n:] + self.Q3nodes[k,n:] + \
        self.Q4nodes[k,n:])
    if min(self.Inodes[k,2,n:]) < 0.:
      self.Inodes[k,2,n:] = np.where(self.Inodes[k,2,n:] < 0., \
        0., self.Inodes[k,2,n:]) # negative fixup
      print 'Negative downward flux fixup performed at node %d' \
          %(k+1)
    if k < self.K-1:
      self.Inodes[k+1,0,n:] = self.Inodes[k,2,n:]
  
  def I_up(self, k):
    '''The discrete ordinate upward equation eq(30).
    '''
    n = self.n/2+self.nup
    self.Inodes[k,0,:n] = self.c[:n]*self.Inodes[k,2,:n] + \
        self.d[:n]*(self.Jnodes[k,:n] + self.Q1nodes[k,:n] + \
        self.Q2nodes[k,:n] + self.Q3nodes[k,:n] + \
        self.Q4nodes[k,:n])
    if min(self.Inodes[k,0,:n]) < 0.:
      self.Inodes[k,0,:n] = np.where(self.Inodes[k,0,:n] < 0., \
         0., self.Inodes[k,0,:n]) # negative fixup
      print 'Negative upward flux fixup performed at node %d' \
          %(k+1)
    if k != 0:
      self.Inodes[k-1,2,:n] = self.Inodes[k,0,:n]

  def reverse(self):
    '''Reverses the transmissivity at soil boundary Myneni 1988b
    eq(41, 42).
    '''
    # check gauss_wt * here....
    # 2 * due to gauss weights summing to 0.5 per hemisphere
    n1 = self.n/2
    n2 = self.n/2+self.nup
    n3 = self.n+self.nup
    Ikz = np.multiply(np.cos(self.views[n2:n3,0]),\
        self.Inodes[self.K-1,2,n2:n3]) # downward view corrected
    self.Inodes[self.K-1,2,:n2] = - 2. * self.refl_s * \
        np.sum(np.multiply(Ikz,self.gauss_wt[n1:])) # reflected

  def converge(self):
    '''Check for convergence and returns true if converges. 
    based on Myneni 1988b eq(44, 45).
    '''
    misclose_top = np.abs((self.Inodes[0,0] - self.Bounds[0])/\
        self.Inodes[0,0])
    misclose_bot = np.abs((self.Inodes[self.K-1,2] - \
        self.Bounds[1])/self.Inodes[self.K-1,2])
    max_top = np.nanmax(misclose_top)
    max_bot = np.nanmax(misclose_bot)
    print 'misclosures top: %.g, and bottom: %.g.' %\
        (max_top, max_bot)
    if max_top  <= self.Tol and max_bot <= self.Tol:
      return True
    else:
      return False
  
  def solve(self):
    '''The solver. Run this as a method of the instance of the
    rt_layers class to solve the RT equations. You first need
    to create an instance of the class by using:
    eg. test = rt_layers() # see rt_layers? for more options.
    then run test.solve().
    Input: none.
    Output: the fluxes at discrete ordinates and nodes.
    '''
    for i in range(self.Iter):
      # forward sweep into the slab
      for k in range(self.K):
        self.I_down(k)
      # reverse the diffuse transmissivity
      self.reverse()
      # backsweep out of the slab
      for k in range(self.K-1,-1,-1):
        self.I_up(k)
      # check for negativity in flux
      if np.min(self.Inodes) < 0.:
        print 'negative values in flux'
      # compute I_k+1/2 and J_k+1/2
      for k in range(self.K):
        self.Inodes[k,1] = (self.Inodes[k,0] + self.Inodes[k,2])/2.
        for j, v in enumerate(self.views):
          self.Jnodes[k,j] = self.J(j,self.Inodes[k,1])
      # acceleration can be implimented here...
      # check for convergence 
      print 'iteration no: %d completed.' % (i+1)
      print self.Inodes[0,0,:self.n/2+self.nup]
      if self.converge() or self.sgl:
        # see Myneni 1989 p 95 for integration of canopy and 
        # soil fluxes below. see also eq(38 - 41).

        I_r_col = self.Inodes[0,0,:self.n/2+self.nup] / \
            -self.mu_s # added /-mu_s 
        I_r_unc_dir = self.I_f(self.views[:self.n/2+self.nup], self.Lc,\
            self.Ir0)
        I_r_unc_dif = self.I_f(self.views[:self.n/2+self.nup], self.Lc,\
            self.Ird)
        I_t_col = self.Inodes[self.K-1,2,\
            self.n/2+self.nup:] / -self.mu_s # added /-mu_s 
        I_t_unc_dir = self.I_f(self.sun0,self.Lc,\
            self.F*self.Beta)
        I_t_unc_dif = self.I_f(self.views[self.n/2+self.nup:],self.Lc,\
            self.F*(1.-self.Beta)) 
        self.I_top_bottom['r_col'] = I_r_col
        self.I_top_bottom['r_unc_dir'] = I_r_unc_dir
        self.I_top_bottom['r_unc_dif'] = I_r_unc_dif 
        self.I_top_bottom['t_col'] = I_t_col
        self.I_top_bottom['t_unc_dir'] = I_t_unc_dir
        self.I_top_bottom['t_unc_dif'] = I_t_unc_dif
        
        print 'solution at iteration %d and saved in class.Inodes.'\
            % (i+1)
        #os.system('play --no-show-progress --null --channels 1 \
        #    synth %s sine %f' % ( 0.5, 500)) # ring the bell
        print 'TOC (up) and soil (down) fluxe array:'
        return self.I_top_bottom
        
      # swap boundary for new flux
      self.Bounds[0] = self.Inodes[0,0]
      self.Bounds[1] = self.Inodes[self.K-1,2]
      continue

  def __del__(self):
    '''The post garbage collection method.
    '''
    print 'An instance of rt_layers has been destroyed.\n'
  
  def __repr__(self):
    '''This prints out the input parameters that define the 
    instance of the class.
    '''
    return \
    '''Tol = %.e, Iter = %i, K = %i, N = %i, Beta = %.3f, Lc = %.3f, 
    refl = %.3f, trans = %.3f, refl_s = %.3f, F = %.4f, sun0_zen = %.3f,
    sun0_azi = %.3f, arch = %s, ln = %.2f, cab = %.2f, car = %.2f, 
    cbrown = %.2f, cw = %.3f, cm = %.3f, lamda = %.0f, CI = %.2f,
    par = %s, Ht = %s, leafd = %s, n = %s k = %s, sgl = %s''' \
        % (self.Tol, self.Iter, self.K, self.N, self.Beta,\
        self.Lc, self.refl, self.trans, self.refl_s, \
        self.F, self.sun0[0]*180./np.pi, self.sun0[1]*180./np.pi,\
        self.arch, self.ln, self.cab, self.car, self.cbrown, \
        self.cw, self.cm, self.lamda, self.CI, self.par, self.Ht, \
        self.leafd, self.spec_n, self.spec_k, self.sgl)

  def I_f(self, view, L, I, hs=1.):
    '''A function that will operate as the Beer's law exponential 
    formula. It requires the angle in radians, the 
    optical depth or LAI, and the initial intensity or flux. The
    example used is in Myneni (19). It  provides the calculation 
    #of the following:
    I_f = I * exp(-abs(G(angle)*hs*L*CI/cos(angle)))
    Input: angle - the illumination zenith angle, L - LAI, I - 
    the intial intensity or flux, hs - hotspot parameter.
    Output: the intensity or flux at L.
    '''
    if np.size(view) > 2:
      angle = view[:,0]
    else:
      angle = view[0]
    mu = np.cos(angle)
    i =  I * np.exp(-abs(G(angle,self.arch,self.par)*hs*L*self.CI/mu))
    return i

  def J(self, index_view, Ia):
    '''The J or Distributed Source Term according to Myneni 1988d
    (21). This gives the multiple scattering as opposed to First 
    Collision term Q.  
    '''
    # check gauss_wt * here...
    # don't divide by 4pi due to integral of gauss_wt is 1 not 4pi...  
    if isinstance(Ia, np.ndarray):
      integ1 = np.multiply(Ia[self.index_gauss],self.Px[index_view])
      integ = np.multiply(integ1,self.Gx[self.index_gauss]/\
          self.Gx[index_view])
      # element-by-element multiplication
      # numerical integration by gaussian qaudrature
      # integral of P2 over hemisphere is 1 due to gauss_wt sum to
      # 1. Thus dont / 4pi
      #j = self.albedo / 4. / np.pi * np.sum(integ*self.gauss_wt)
      # used /2. as in the one angle case.
      j = self.albedo * np.sum(integ*self.gauss_wt) / 2.# / 2. 
    else:
      raise Exception('ArrayInputRequired')
    return j

  def Q1(self, index_view, k):
    '''The Q1 First First Collision Source Term as defined in Myneni
    1988d (16). This is the downwelling direct part of the Q term. 
    '''
    L = self.mid_ks[k]
    # check * 4 here...  
    I = self.I_f(self.sun0, L, self.I0, self.hspot[k, index_view])
    #q = self.albedo / 4. * self.Ps[index_view] * self.Gs/\
    #    self.Gx[index_view] * I
    # don't need to /4/pi due to P2 summing to 1 not 4pi all
    # due to wts.
    q = self.albedo / 4. * I * self.Ps[index_view] * self.Gs/\
        self.Gx[index_view]
    return q

  def Q2(self, index_view, k):
    '''The Q2 Second First Collision Source Term as defined in
    Myneni 1988d (17). This is the downwelling diffuse part of 
    the Q term.
    '''
    L = self.mid_ks[k]
    # check gauss_wt * here...  
    integ1 = np.multiply(self.Px[index_view, self.n/2:], 
        self.Gx[self.n/2+self.nup:self.n+self.nup]/\
        self.Gx[index_view])
    integ = np.multiply(integ1, self.I_f(self.sun_down[:self.n/2], L, \
        self.Id))
    #q = self.albedo / 4. * np.sum(np.multiply(integ,\
    #    self.gauss_wt[self.n/2:]))
    q = self.albedo / 4. * np.sum(np.multiply(integ,\
        self.gauss_wt[self.n/2:] * 2.)) # due to hemisphere sum 1.
    return q

  def Q3(self, index_view, k): 
    '''The Q3 Third First Collision Source Term as defined in
    Myneni 1988d (18). This is the upwelling direct part of 
    the Q term.
    '''
    # check gauss_wt * here...
    # due to upper hemisphere only being 2pi divide only by 2 
    dL = self.mid_ks[k]
    dL = self.Lc - dL
    integ1 = np.multiply(self.Px[index_view,:self.n/2],\
        self.Gx[:self.n/2]/self.Gx[index_view]) 
        # element-by-element multipl.
    integ = np.multiply(integ1, self.I_f(self.sun_up[:self.n/2], -dL,\
        self.Ir0)) # ^^
    #q = self.albedo / 4. * np.sum(np.multiply(integ, \
    #    self.gauss_wt[:self.n/2]))
    # numerical integration by gaussian quadrature
    q = self.albedo / 4. * np.sum(np.multiply(integ, \
        self.gauss_wt[:self.n/2] * 2.)) # due to hemisphere sum 1.
    return q

  def Q4(self, index_view, k):
    '''The Q4 Fourth First Collision Source Term as defined in 
    Myneni 1988d (19). The is the upwelling diffuse part of
    the Q term.
    '''
    # check gauss_wt * here...
    # due to upper hemisphere only being 2pi divide only by 2
    dL = self.mid_ks[k]
    dL = self.Lc - dL
    integ1 = np.multiply(self.Px[index_view,:self.n/2],\
        self.Gx[:self.n/2]/self.Gx[index_view])
    integ = np.multiply(integ1, self.I_f(self.sun_up[:self.n/2], -dL,\
        self.Ird))
    #q = self.albedo / 4. * np.sum(np.multiply(integ, \
    #    self.gauss_wt[:self.n/2]))
    q = self.albedo / 4. * np.sum(np.multiply(integ, \
        self.gauss_wt[:self.n/2] * 2.))
    return q
    
  def Scalar_flux(self):
    '''A method to return the scalar fluxes of canopy reflection,
    collided soil absorption, uncollided soil absorption, and canopy 
    absorption. Based on eq(38 - 41).  
    Input: none.
    Output: canopy refl, collided soil absorption, uncollided soil
    absorption, canopy absorption.
    '''
    # check gauss_wt * here...
    r_col = self.I_top_bottom['r_col'][:self.n/2]
    r_unc_dir = self.I_top_bottom['r_unc_dir'][:self.n/2]
    r_unc_dif = self.I_top_bottom['r_unc_dif'][:self.n/2]
    t_col = self.I_top_bottom['t_col'][:self.n/2]
    t_unc_dir = self.I_top_bottom['t_unc_dir']
    t_unc_dif = self.I_top_bottom['t_unc_dif'][:self.n/2]

    c_refl = np.sum(np.multiply(r_col + r_unc_dir + \
        r_unc_dif, self.gauss_wt[:self.n/2]) * 2.) 
        # I hate doing this... * 2 due to 
    # 2 * gauss_wt for hemisphere sum to 1.
    s_abs_col = np.sum(np.multiply(t_col, self.gauss_wt[self.n/2:] \
        * 2)) * (1. - self.refl_s)# this... 
    s_abs_unc = (np.sum(np.multiply(t_unc_dif, self.gauss_wt[self.n/2:] \
        * 2.)) + t_unc_dir) * (1. - self.refl_s) 
    c_abs = 1. - c_refl - s_abs_col - s_abs_unc
    return (c_refl,s_abs_col,s_abs_unc,c_abs)

def plot_sphere(obj):
  '''A function that plots the full RF over the upper and lower
  hemispheres. 
  Input: rt_layers object.
  Output: spherical plot of rf.
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  sunz = np.cos(obj.sun0[0])
  suny = np.cos(obj.sun0[1]) * np.sqrt(1. - sunz**2)
  sunx = np.sin(obj.sun0[1]) * np.sqrt(1. - sunz**2)
  z = np.cos(obj.views[:,0])
  y = np.cos(obj.views[:,1]) * np.sqrt(1. - z**2) # swap for NE
  x = np.sin(obj.views[:,1]) * np.sqrt(1. - z**2)
  c = np.concatenate((obj.I_top_bottom['r_col'] + \
      obj.I_top_bottom['r_unc_dir'] + obj.I_top_bottom['r_unc_dif'],\
      obj.I_top_bottom['t_col'] + obj.I_top_bottom['t_unc_dif']\
      + obj.I_top_bottom['t_unc_dir']))
  scat = ax.scatter(x, y, z, c=c)
  ax.plot((-sunx, sunx), (-suny, suny), (-sunz, sunz), 'r--') 
  ax.scatter(-sunx, -suny, -sunz, s=100, c='r')   # sun location
  ax.set_xlabel('E axis')
  ax.set_ylabel('N axis')
  ax.set_zlabel('Z\' axis')
  plt.title('RF over the sphere')
  plt.colorbar(scat, shrink=0.5, aspect=10)
  plt.show()

def plot_contours(obj, top_bottom=True):
  '''A function that plots the RF as an azimuthal projection
  with contours over the TOC and soil.
  Input: rt_layers object, top_bottom - True if only TOC plot, False
  if both TOC and soil.
  Output: contour plot of brf.
  '''
  sun = ((np.pi - obj.sun0[0]) * np.sin(obj.sun0[1] + np.pi), \
     (np.pi - obj.sun0[0]) * np.cos(obj.sun0[1] + np.pi)) 
  theta = obj.views[:,0]
  theta[obj.n/2+obj.nup:] = np.abs(theta[obj.n/2+obj.nup:] - np.pi)
  y = np.cos(obj.views[:,1]) * theta # swap xy for NE convention
  x = np.sin(obj.views[:,1]) * theta
  c_refl = obj.I_top_bottom['r_col'] + obj.I_top_bottom['r_unc_dir']\
      + obj.I_top_bottom['r_unc_dif']
  c_trans = (obj.I_top_bottom['t_col'] + obj.I_top_bottom['t_unc_dif']\
      + obj.I_top_bottom['t_unc_dir'])
  z = np.append(c_refl, c_trans)
  if top_bottom == True:
    if maxz > 1.:
      maxz = np.max(z)
    else:
      maxz = 1.
  else:
    maxz = np.max(z[:obj.n/2+obj.nup])
  #maxz = np.max(z) 
  minz = 0. #np.min(z)
  space = np.linspace(minz, maxz, 11)
  xt = x[:obj.n/2+obj.nup]
  yt = y[:obj.n/2+obj.nup]
  xb = x[obj.n/2+obj.nup:]
  yb = y[obj.n/2+obj.nup:]
  zt = z[:obj.n/2+obj.nup]
  zb = z[obj.n/2+obj.nup:]
  fig = plt.figure()
  if top_bottom == True:
    plt.subplot(121)
  plt.plot(sun[0], sun[1], 'ro')
  triangt = tri.Triangulation(xt, yt)
  plt.gca().set_aspect('equal')
  plt.tricontourf(triangt, zt, space, vmax=maxz, vmin=minz)
  plt.title('TOC reflectance')
  plt.ylabel('N')
  plt.xlabel('E')
  if top_bottom == True:
    plt.subplot(122)
    plt.plot(-sun[0], -sun[1], 'ro')
    plt.gca().set_aspect('equal')
    triangb = tri.Triangulation(xb, yb)
    plt.tricontourf(triangb, zb, space, vmax=maxz, vmin=minz)
    plt.title('BOC transmittance')
    #plt.ylabel('Y')
    plt.xlabel('E')
  s = obj.__repr__()
  if top_bottom == True:
    cbaxes = fig.add_axes([0.11,0.1,0.85,0.05])
    plt.suptitle(s,x=0.5,y=0.93)
    plt.colorbar(orientation='horizontal', ticks=space,\
      cax = cbaxes, format='%.3f')
  else:
    plt.suptitle(s,x=0.5,y=0.13)
    plt.colorbar(orientation='horizontal', ticks=space,\
        format='%.3f')
    #plt.tight_layout() 
  plt.show()

def xsect_array(azi=0., top_bot='Top', steps=20, sun=None):
  '''A function that returns an array that could be used as user
  input as <user_views> for cross-sections plots. If sun is 
  included azi should be the incoming solar direction.
  Input: azi - the azimuth for the cross-section. top_bot -
  'Top' for reflectance, 'Bot' for transmittance. - steps - 
  mininum points along the azimuth to sample. sun - the solar 
  zenith if required.
  Output: array in [zen, azi] format.
  '''
  if top_bot=='Top':
    zen = np.linspace(-85., 85., steps, endpoint=True) 
    #zen[0] += 0.001
    #zen[-1] -= 0.001
    azia = np.where(zen < 0., azi+180., azi)
    if sun is not None:
      sun = np.abs(sun-180.)
      if sun not in zen: 
        index = np.searchsorted(zen, -sun)
        azia = np.insert(azia, index, azi+180.)
        zen = np.insert(zen, index, sun)
    zen = np.abs(zen) 
  if top_bot=='Bot':
    steps += 1
    steps = (steps // 2) * 2
    zen = np.concatenate((np.linspace(95., 180., steps/2, \
        endpoint=True), np.linspace(180., 95., steps/2, \
        endpoint=True)[1:])) 
    azia = np.ones(steps-1)
    azia[:steps/2] = azi
    azia[steps/2:] = azi+180.
    if sun is not None and sun not in zen: 
      index = np.searchsorted(zen, sun)
      azia = np.insert(azia, index, azi)
      zen = np.insert(zen, index, sun)
  azi = azia
  arr = np.dstack((zen, azi))
  return arr[0]

def plot_xsect(obj, top_bottom=True):
  '''A function that plots a cross section of reflectance and/or
  transmittance. This requires the user input <user_views> to be
  created by the obj.xsect_array() function. 
  Input: obj - an instance of the two_angle class which has been
  solved for. top_bottom - False means only the reflectance will
  be plotted, True means both reflectance and transmittance will
  be plotted.
  Output: A plot of the cross section/s.
  '''
  zens = obj.views_zen.copy()*180./np.pi
  c_refl = obj.I_top_bottom['r_col'] + obj.I_top_bottom['r_unc_dir']\
     + obj.I_top_bottom['r_unc_dif']
  c_refl = c_refl[obj.n/2:] 
  max_refl = np.max(c_refl)
  min_refl = np.min(c_refl)
  zens_top = zens[obj.n/2:obj.n/2+obj.nup]
  zens_top[:obj.nup/2+1] = -zens_top[:obj.nup/2+1]
  xticks = np.array([-90.,-60.,-30.,0.,30.,60.,90.])
  if top_bottom==True: 
    c_trans = (obj.I_top_bottom['t_col'] + obj.I_top_bottom['t_unc_dif']\
        + obj.I_top_bottom['t_unc_dir'])
    c_trans = c_trans[obj.n/2:] 
    max_trans = np.max(c_trans)
    min_trans = np.min(c_trans)
    max_y = np.max([max_refl, max_trans])*1.1
    min_y = np.min([min_refl, min_trans])*0.9
    zens_bot = zens[obj.n+obj.nup:] - 180.
    zens_bot[obj.nup/2:] = -zens_bot[obj.nup/2:]
    plt.subplot(121)
  else:
    max_y = max_refl*1.1
    min_y = min_refl*0.9
  plt.plot(zens_top, c_refl, 'b-', label='TOC reflectance')
  plt.ylim((min_y,max_y))
  plt.xlim((-90.,90))
  plt.xticks(xticks)
  plt.title('TOC reflectance')
  plt.ylabel('Reflectanc')
  plt.xlabel('View Zenith')
  if top_bottom==True:
    plt.subplot(122) 
    plt.plot(zens_bot, c_trans, 'g-.', label='BOC transmittance')
    plt.ylim((min_y,max_y))
    plt.xlim((-90.,90.))
    plt.xticks(xticks)
    plt.title('BOC transmittance')
    plt.ylabel('Transmittance')
    plt.xlabel('View Zenith')
  plt.tight_layout()
  plt.show()

