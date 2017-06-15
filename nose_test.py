#!/usr/bin/python

''' A script that tests the one_angle.py module.
    Run by: nosetests -v test_one_angle.py
'''

import one_angle as oa
import two_angle as ta
import radtran as rt
import numpy as np
from scipy.integrate import *
import pickle
import pdb

# loading the quadrrature set for calculations
# spherical quadrature
N = 8 
f = open('quad_dict.dat')
quadr_dict = pickle.load(f)
f.close()
quadr = quadr_dict[str(N)]
gauss_wt2 = quadr[:,2] * 4. * np.pi / 8. # due to octants
gauss_vw2 = np.array(zip(quadr[:,0],quadr[:,1])) 
# linear quadrature
gauss_f_mu = open('lgvalues-abscissa.dat','rb')
gauss_f_wt = open('lgvalues-weights.dat','rb')
gauss_mu1 = pickle.load(gauss_f_mu)
gauss_wt1 = pickle.load(gauss_f_wt)
# sort all dictionary items
for k in gauss_mu1.keys():
  ml = gauss_mu1[k]
  wl = gauss_wt1[k]
  ml, wl = zip(*sorted(zip(ml,wl),reverse=True))
  gauss_mu1[k] = ml
  gauss_wt1[k] = wl
N = 16
gauss_mu1 = gauss_mu1[str(N)]
gauss_wt1 = gauss_wt1[str(N)]


def test_gl():
  '''A function to test to integration of the leaf angle 
  distribution function gl() between 0 and pi/2 is unity.
  '''
  arch = ['p','e','s','m','x','u','l']
  g = []
  for a in arch:
    g.append(quad(rt.gl, 0., np.pi/2., args=(a))[0])
  np.testing.assert_almost_equal(g, 1., decimal=6)
  #return g

def test_one_angle_fluxes():
  '''A function that tests one_angle.py flux calculation.
  see table in Myneni 1988b for test data.
  '''
  test = oa.rt_layers(Tol=1.e-06,Iter=200,K=40,N=16,\
      Lc=4.,refl=0.175,trans=0.175,refl_s=0.2,I0=1.,\
      sun0=120.,arch='s')
  test.solve()
  c,s,a = test.Scalar_flux()
  val = np.array([c,s,a])
  truth = np.array([0.0988, 0.0384,\
      0.8628])
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)

def test_two_angle_fluxes():
  '''A function that tests two_angle.py flux calculation.
  see table in Myneni p.95 for test data. Compared against
  Odom data in Myneni 1988d p.110.  '''
  test = ta.rt_layers(Tol=1.e-06,Iter=200,K=20,N=4,\
      Lc=2.,refl=1.,trans=0.,refl_s=0.,sun0_zen=120.,\
      F=1,sun0_azi=.0,arch='s',Beta=1.0)
  test.solve()
  c,sc,su,a = test.Scalar_flux()
  val = np.array([c,sc,su,a])
  truth = np.array([0.498, 0.366, 0.135, 0.])
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)
  
def test_G():
  '''A function to test the G-projection function for integration
  to 1. between 0 an pi for the one angle case. Myneni 1988a
  eq(17)''' 
  truth = 1
  arch = ['p','e','s','m','x','u']
  vals = []
  for a in arch:
    Gg = []
    for mu, wt in zip(gauss_mu1, gauss_wt1):
      Gg.append(rt.G(np.arccos(mu), a))
    val = np.sum(np.multiply(Gg, gauss_wt1))
    vals.append(val) 
  #return (vals, truth)
  np.testing.assert_almost_equal(vals, truth, decimal=2)

def test_psi_Big_psi():
  '''A function that test the one angle Big psi function against
  the product of two psi functions if trans == refl. See Myneni
  1988a eq(42 and 43)'''
  suns = np.linspace((0 + 1.) * np.pi / 180., (180. - 1) \
      * np.pi / 180)
  view = np.pi/4.
  leaf = np.pi 
  truth = []
  vals = []
  for sun in suns:
    truth.append(rt.psi(leaf, view) * rt.psi(leaf, sun))
    vals.append(rt.Big_psi(view, sun, leaf, 'r') + \
        rt.Big_psi(view, sun, leaf, 't'))
  # return(val, truth)
  np.testing.assert_almost_equal(vals, truth, decimal=2)

def test_Gamma_unity():
  '''A function that tests the one angle Gamma phase function for 
  integration to unity in combination with G and albedo. See 
  Myneni 1988a eq(9).'''
  sun = 180./180. * np.pi
  arch = 's'
  refl = .5
  trans = .5
  albedo = refl + trans 
  truth = 1.
  val = []
  for mu in gauss_mu1:
    val.append(rt.Gamma(np.arccos(mu), sun, arch, refl, trans)\
        /albedo/rt.G(sun, arch) / np.pi)
  val = np.sum(np.multiply(val, gauss_wt1)) * 2. * np.pi # hate this
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=1)

def test_dot():
  '''A function to test the dot product function used to 
  calculate the cosine between 2 spherical angles.
  The dot product is used in the calculation of the
  Big_psi2 function.
  '''
  sun = (180. * np.pi / 180., 45. * np.pi / 180.)
  view = (90. * np.pi / 180., 45. * np.pi / 180.)
  truth = np.cos(np.pi/2)
  val = rt.dot(sun, view)
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=6)
 
def test_Big_psi2():
  '''A function to test the radtran.py Big_psi2 function against a 
  dot integrated function using scipy integration solution based
  on Myneni 1988c eq(13).
  '''
  leaf_ze = 82. * np.pi / 180.
  view_ze = 25. * np.pi / 180.
  view_az = 63. * np.pi / 180.
  view = (view_ze, view_az)
  sun_ze = 74. * np.pi / 180.
  sun_az = 29. * np.pi / 180.
  sun = (sun_ze, sun_az)
  val = rt.Big_psi2(view, sun, leaf_ze)
  val = val[0] - val[1]
  def integ(leaf_az, leaf_ze, view_ze, view_az, sun_ze, sun_az):
    view = (view_ze, view_az)
    sun = (sun_ze, sun_az)
    dots = np.array([])
    for la in leaf_az:
      leaf = (leaf_ze, la)
      sun = (sun_ze, sun_az)
      view = (view_ze, view_az)
      dots = np.append(dots, rt.dot(leaf, sun) * rt.dot(leaf, view))
    return dots
  truth = fixed_quad(integ, 0, 2.*np.pi, args=(leaf_ze, view_ze,\
      view_az, sun_ze, sun_az))[0] / np.pi / 2.
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=3)

def test_Gamma2_unity():
  '''A function that tests the radtran.py Gamma2 phase function
  wrt integration to unity. Based on Myneni 1988c eq(6).'''
  sun = (-180./180. * np.pi, 90./180. * np.pi)
  arch = 'u'
  refl = 1.
  trans = 0.
  albedo = refl + trans 
  truth = 1.
  val = []
  for v in gauss_vw2:
    val.append(rt.Gamma2(v, sun, arch, refl, trans, 1.)\
        /albedo/rt.G(sun[0], arch) / np.pi)
  val = np.sum(np.multiply(val, gauss_wt2))
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)

def test_Gamma2_P_G_albedo():
  '''A function that tests the radtran.py Gamma2 phase function
  wrt P and G functions and albedo. Based on Myneni 1988c eq(7).'''
  view = (120./180. * np.pi, 45./180. * np.pi)
  sun = (-100./180. * np.pi, 90./180. * np.pi)
  arch = 'u'
  refl = .5
  trans = 0.5
  albedo = refl + trans 
  P = rt.P2(view, sun, arch, refl, trans, 1.)
  G = rt.G(sun[0], arch)
  truth = P * albedo * G / 4.
  val = rt.Gamma2(view, sun, arch, refl, trans, 1.)
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=3)

def test_Gamma2():
  '''A function that tests the radtran.py Gamma2 phase function
  for integration to albedo * G(sun) over the complete sphere.
  Based on Myneni 1988c eq(2).
  '''
  sun = (-180./180. * np.pi, 90./180. * np.pi)
  arch = 'u'
  refl = 1.
  trans = 0.
  albedo = refl + trans 
  truth = albedo * rt.G(sun[0],arch)
  Gam = []
  for v in gauss_vw2:
    Gam.append(rt.Gamma2(v, sun, arch, refl, trans, 1.) /\
        np.pi)
  val = np.sum(np.multiply(Gam, gauss_wt2))
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)

def test_Psi2_psi():
  '''A function that test the 2 and 1 angle Big_psi functions 
  against each other. Based on Myneni 1988c (19).
  '''
  view_zen = 40.*np.pi/180.
  sun_zen = 15.*np.pi/180.
  sun_azi = 60.*np.pi/180.
  leaf_zen = 20.*np.pi/180.
  arch = 'u'
  refl = .0
  trans = 1.
  truth = rt.Big_psi(view_zen, sun_zen, leaf_zen, 't')
  def fun(view_azi, view_zen, sun_azi, sun_zen, leaf_zen,\
      refl, trans):
    view = (view_zen, view_azi)
    sun = (sun_zen, sun_azi)
    f = rt.Big_psi2(view, sun, leaf_zen)
    f = f[0]*trans + f[1]*refl
    return f
  val = quad(fun, 0., 2.*np.pi, args=(view_zen, sun_azi, \
      sun_zen, leaf_zen, refl, trans))[0] / np.pi / 2.
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=1)

def test_P2():
  '''A function that tests the radtran.py P2 phase function for
  integration to unity over the complete sphere.
  '''
  sun = (0. * np.pi / 180. , 45.) # zenith, azimuth in radians
  arch = 'u'
  refl = 0.5
  trans = 0.5
  truth = 1.0
  val = []
  for v in gauss_vw2:
    val.append(rt.P2(v, sun, arch, refl, trans, 1.))
  val = np.sum(np.multiply(val, gauss_wt2)) / (4.*np.pi)
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=2)

def test_P_unity():
  '''A function that tests the one angle P phase function for 
  integration to unity. See Myneni 1988a eq(10).'''
  sun = 180./180. * np.pi
  arch = 's'
  refl = .5
  trans = .5
  albedo = refl + trans 
  truth = 1.
  val = []
  for mu in gauss_mu1:
    val.append(rt.P(np.arccos(mu), sun, arch, refl, trans)) 
  val = np.sum(np.multiply(val, gauss_wt1)) / 2. # I hate this...
  #return (val, truth)
  np.testing.assert_almost_equal(val, truth, decimal=1)


