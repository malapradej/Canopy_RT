#!/usr/bin/python
'''A script that creates the files required for the rami romc
online radiative transfer model intercomparison. 
'''
import numpy as np
import os
import two_angle as ta
import multiprocessing
import pdb

# no of cpus to use
cpus = 2

exps = [ 'HOM01_TUR_ERE_NR1_00', 'HOM01_TUR_ERE_NR1_30', 'HOM01_TUR_ERE_NR1_60',\
'HOM01_TUR_PLA_NR1_00', 'HOM01_TUR_PLA_NR1_30', 'HOM01_TUR_PLA_NR1_60',\
'HOM02_TUR_ERE_NR1_00', 'HOM02_TUR_ERE_NR1_30', 'HOM02_TUR_ERE_NR1_60',\
'HOM02_TUR_PLA_NR1_00', 'HOM02_TUR_PLA_NR1_30', 'HOM02_TUR_PLA_NR1_60',\
'HOM03_DIS_ERE_NIR_20', 'HOM03_DIS_ERE_NIR_50', 'HOM03_DIS_ERE_RED_20',\
'HOM03_DIS_ERE_RED_50', 'HOM03_TUR_PLA_NIR_20', 'HOM03_TUR_PLA_NIR_50',\
'HOM03_TUR_PLA_RED_20', 'HOM03_TUR_PLA_RED_50', 'HOM03_TUR_UNI_NIR_20',\
'HOM03_TUR_UNI_NIR_50', 'HOM03_TUR_UNI_RED_20', 'HOM03_TUR_UNI_RED_50',\
'HOM05_TUR_ERE_NR1_00', 'HOM05_TUR_ERE_NR1_30', 'HOM05_TUR_ERE_NR1_60',\
'HOM05_TUR_PLA_NR1_00', 'HOM05_TUR_PLA_NR1_30', 'HOM05_TUR_PLA_NR1_60',\
'HOM11_DIS_ERE_NR1_00', 'HOM11_DIS_ERE_NR1_30', 'HOM11_DIS_ERE_NR1_60',\
'HOM11_DIS_PLA_NR1_00', 'HOM11_DIS_PLA_NR1_30', 'HOM11_DIS_PLA_NR1_60',\
'HOM12_DIS_ERE_NR1_00', 'HOM12_DIS_ERE_NR1_30', 'HOM12_DIS_ERE_NR1_60',\
'HOM12_DIS_PLA_NR1_00', 'HOM12_DIS_PLA_NR1_30', 'HOM12_DIS_PLA_NR1_60',\
'HOM13_DIS_PLA_NIR_20', 'HOM13_DIS_PLA_NIR_50', 'HOM13_DIS_PLA_RED_20',\
'HOM13_DIS_PLA_RED_50', 'HOM15_DIS_ERE_NR1_00', 'HOM15_DIS_ERE_NR1_30',\
'HOM15_DIS_ERE_NR1_60', 'HOM15_DIS_PLA_NR1_00', 'HOM15_DIS_PLA_NR1_30',\
'HOM15_DIS_PLA_NR1_60' ]

canopy = {'HOM01' : {'Lc' : 1., 'Ht' : 1.}, 'HOM02' : {'Lc' : 2., 'Ht' :\
    1.}, 'HOM03' : {'Lc' : 3., 'Ht' : 2.}, 'HOM05' : {'Lc' : 5., 'Ht' :\
    1.}, 'HOM11' : {'Lc' : 1., 'Ht' : 1.}, 'HOM12' : {'Lc' : 2., 'Ht' :\
    1.}, 'HOM13' : {'Lc' : 3., 'Ht' : 2.}, 'HOM15' : {'Lc' : 5., 'Ht' :\
    1.}}

leaf = {'TUR' : None, 'DIS' : 0.1}

scat = {'NR1' : {'refl_s' : 1., 'refl' : 0.5, 'trans' : 0.5}, \
    'NIR' : {'refl_s' : 0.159, 'refl' : 0.4957, 'trans' : 0.4409}, \
    'RED' : {'refl_s' : 0.127, 'refl' : 0.0546, 'trans' : 0.0149}}

vzas = np.arange(-75., 76., 2)
vazs_op = np.where(vzas<0., 90., 270.)
vazs_pp = np.where(vzas<0., 0., 180.)
vzas = np.abs(vzas)
sun0_azi = 0.
sun0_zens = {'00' : 180., '20' : 160., '30' : 150., '50' : 130.,\
    '60' : 120.} 
user_views = np.vstack((np.concatenate((vzas,vzas)),np.concatenate(\
    (vazs_op,vazs_pp)))).T
user_views_rad = user_views * np.pi / 180.

archs = {'PLA' : 'p', 'ERE' : 'e', 'UNI' : 'u'}

path = 'rami_debug'
if not(os.path.exists(path)):
  os.mkdir(path)

model = 'dofin2'

flog = open(os.path.join(path, 'run.log'), 'w')
place = '-1.000000'

def mp_worker(exp):
#for exp in exps:
  Lc = canopy[exp[:5]]['Lc']
  Ht = canopy[exp[:5]]['Ht']
  leafd = leaf[exp[6:9]]
  arch = archs[exp[10:13]]
  refl_s = scat[exp[14:17]]['refl_s']
  refl = scat[exp[14:17]]['refl']
  trans = scat[exp[14:17]]['trans']
  sun0_zen = sun0_zens[exp[18:]]
  sun0_zen_rad = np.abs((sun0_zen - 180.) * np.pi / 180.)
  
  # the instance of all reflectances (single and multiple)
  exp_obj = ta.rt_layers(Lc=Lc, Ht=Ht, leafd=leafd, arch=arch, \
      refl_s=refl_s, refl=refl, trans=trans, sun0_zen=sun0_zen,\
      user_views=user_views)
  exp_obj.solve()
  flog.write('%s\n' % exp)
  flog.write('%s\n' % exp_obj)
  
  flux = exp_obj.Scalar_flux()
  flog.write('%s\n' % str(flux)) 
  I_top_bottom = exp_obj.I_top_bottom
  n = exp_obj.n
  nup = exp_obj.nup
  ndn = exp_obj.ndn
  gauss_wt = exp_obj.gauss_wt 
  
  # brf1 
  brf1 = flux[0]
  fname = 'brf1_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname)
  fl = open(fname, 'w')
  fl.write('1\t5\t%s\n' % place) 
  fl.write('%.6f\t%s\t%s\t%.6f\t%s\n' % (sun0_zen_rad, place, \
      place, brf1, place))
  fl.close()
  
  #fabs
  fabs = '%.6f\n' % (flux[3] * 100.)
  fname = 'fabs_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname)
  fl = open(fname, 'w')
  fl.write(fabs)
  fl.close() 

  t_col = I_top_bottom['t_col']
  t_unc_dif = I_top_bottom['t_unc_dif']
  t_unc_dir = I_top_bottom['t_unc_dir']
  
  #ftran
  ftran = (np.sum(np.multiply(t_col[:n/2], gauss_wt[n/2:] * 2))\
      + (np.sum(np.multiply(t_unc_dif[:n/2], gauss_wt[n/2:] * 2.)))\
      + t_unc_dir) * 100.
  ftran = '%.6f\n' % ftran
  fname = 'ftran_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname)
  fl = open(fname, 'w')
  fl.write(ftran)
  fl.close()

  r_col = I_top_bottom['r_col']
  r_unc_dif = I_top_bottom['r_unc_dif']
  r_unc_dir = I_top_bottom['r_unc_dir']

  #brfop
  brfop_ = r_col[n/2:n/2+nup/2] + r_unc_dif[n/2:n/2+nup/2] + \
      r_unc_dir[n/2:n/2+nup/2] 
  brfop = np.insert(user_views_rad[:nup/2], 2, brfop_, axis=1)
  brfop = np.insert(brfop, 0, sun0_zen_rad, axis=1)
  brfop = np.insert(brfop, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfop.shape[0], brfop.shape[1], place)
  fname = 'brfop_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfop, fmt='%.6f', delimiter='\t', header=header,\
      comments='')

  #brfpp
  brfpp_ = r_col[n/2+nup/2:] + r_unc_dif[n/2+nup/2:] + \
      r_unc_dir[n/2+nup/2:] 
  brfpp = np.insert(user_views_rad[nup/2:], 2, brfpp_, axis=1)
  brfpp = np.insert(brfpp, 0, sun0_zen_rad, axis=1)
  brfpp = np.insert(brfpp, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfpp.shape[0], brfpp.shape[1], place)
  fname = 'brfpp_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfpp, fmt='%.6f', delimiter='\t', header=header,\
      comments='')
  
  # the instance of only single scattering 
  exp_obj = ta.rt_layers(Lc=Lc, Ht=Ht, leafd=leafd, arch=arch, \
      refl_s=refl_s, refl=refl, trans=trans, sun0_zen=sun0_zen,\
      user_views=user_views, sgl=True)
  exp_obj.solve()
  flog.write('%s\n' % exp)
  flog.write('%s\n' % exp_obj)
  
  I_top_bottom = exp_obj.I_top_bottom
  n = exp_obj.n
  nup = exp_obj.nup
  ndn = exp_obj.ndn
  gauss_wt = exp_obj.gauss_wt 
  
  r_col = I_top_bottom['r_col']
  r_unc_dif = I_top_bottom['r_unc_dif']
  r_unc_dir = I_top_bottom['r_unc_dir']

  #brfop_co_sgl
  brfop_co_sgl_ = r_col[n/2:n/2+nup/2] 
  brfop_co_sgl = np.insert(user_views_rad[:nup/2], 2, brfop_co_sgl_, axis=1)
  brfop_co_sgl = np.insert(brfop_co_sgl, 0, sun0_zen_rad, axis=1)
  brfop_co_sgl = np.insert(brfop_co_sgl, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfop_co_sgl.shape[0], brfop_co_sgl.shape[1],\
      place)
  fname = 'brfop_co_sgl_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfop_co_sgl, fmt='%.6f', delimiter='\t',\
      header=header, comments='')

  #brfpp_co_sgl
  brfpp_co_sgl_ = r_col[n/2+nup/2:] 
  brfpp_co_sgl = np.insert(user_views_rad[nup/2:], 2, brfpp_co_sgl_, axis=1)
  brfpp_co_sgl = np.insert(brfpp_co_sgl, 0, sun0_zen_rad, axis=1)
  brfpp_co_sgl = np.insert(brfpp_co_sgl, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfpp_co_sgl.shape[0], brfpp_co_sgl.shape[1],\
      place)
  fname = 'brfpp_co_sgl_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfpp_co_sgl, fmt='%.6f', delimiter='\t', \
      header=header, comments='')

  #brfop_uc_sgl
  brfop_uc_sgl_ = r_unc_dif[n/2:n/2+nup/2] + r_unc_dir[n/2:n/2+nup/2]
  brfop_uc_sgl = np.insert(user_views_rad[:nup/2], 2, brfop_uc_sgl_, axis=1)
  brfop_uc_sgl = np.insert(brfop_uc_sgl, 0, sun0_zen_rad, axis=1)
  brfop_uc_sgl = np.insert(brfop_uc_sgl, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfop_uc_sgl.shape[0], brfop_uc_sgl.shape[1],\
      place)
  fname = 'brfop_uc_sgl_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfop_uc_sgl, fmt='%.6f', delimiter='\t',\
      header=header, comments='')

  #brfpp_uc_sgl
  brfpp_uc_sgl_ = r_unc_dif[n/2+nup/2:] + r_unc_dir[n/2+nup/2:]
  brfpp_uc_sgl = np.insert(user_views_rad[nup/2:], 2, brfpp_uc_sgl_, axis=1)
  brfpp_uc_sgl = np.insert(brfpp_uc_sgl, 0, sun0_zen_rad, axis=1)
  brfpp_uc_sgl = np.insert(brfpp_uc_sgl, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfpp_uc_sgl.shape[0], brfpp_uc_sgl.shape[1],\
      place)
  fname = 'brfpp_uc_sgl_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfpp_uc_sgl, fmt='%.6f', delimiter='\t', \
      header=header,\
      comments='') 

  #brfop_mlt
  brfop_mlt = brfop_ - brfop_co_sgl_ - brfop_uc_sgl_
  brfop_mlt = np.insert(user_views_rad[:nup/2], 2, brfop_mlt, axis=1)
  brfop_mlt = np.insert(brfop_mlt, 0, sun0_zen_rad, axis=1)
  brfop_mlt = np.insert(brfop_mlt, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfop_mlt.shape[0], brfop_mlt.shape[1], place)
  fname = 'brfop_mlt_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfop_mlt, fmt='%.6f', delimiter='\t', header=header,\
      comments='')

  #brfpp_mlt
  brfpp_mlt = brfpp_ - brfpp_co_sgl_ - brfpp_uc_sgl_
  brfpp_mlt = np.insert(user_views_rad[nup/2:], 2, brfpp_mlt, axis=1)
  brfpp_mlt = np.insert(brfpp_mlt, 0, sun0_zen_rad, axis=1)
  brfpp_mlt = np.insert(brfpp_mlt, 4, -1, axis=1)
  header = '%i\t%i\t%s' % (brfpp_mlt.shape[0], brfpp_mlt.shape[1], place)
  fname = 'brfpp_mlt_%s-%s.mes' % (model, exp)
  fname = os.path.join(path,fname) 
  np.savetxt(fname, brfpp_mlt, fmt='%.6f', delimiter='\t', header=header,\
      comments='') 

  flog.flush()

def mp_handler():
  p = multiprocessing.Pool(cpus)
  p.map(mp_worker, exps)

if __name__ == '__main__':
  mp_handler() 

flog.close()

