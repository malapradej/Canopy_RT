This is a Radiative Transfer Model project to describe the influence of vegetation structure on the emitted TOC SIF signal.

There are 3 RT models in the project:
1) one_angle.py contains the one angle 1D model as described in Myneni et al 1988a and b
2) two_angle.py contains the two angle 1D model as described in Myneni et al 1988c and d.
3) ThreeD_angle.py is the 3D model based on Myneni 1991. 
All the above models are based on the Discrete Ordinates Finite Differencing method. 
The majority of the RT functions which are similar to all models is found in radtran.py. 
The nose_test.py file contains all the tests on the function and results of calculations of the models. 
The code has been commented as much as possible to improve understanding, although arguably there will always be a preference for more comments depending on familiarity with python coding.
There are also various input files used in the quadrature integration calculations such as all files containing quad in their names as well the lgvalues files. 
The lad_model_quad.py file is used to build example 3D datasets of trees used in the 3D model. 
To use the PROSPECT leaf model requires the installation of f2py and running the 'compil_f2py.sh' script. This creates the prospect_py.so file specific to the PC architecture used. This is gives access to the fortran code from python.

To run the code in python:
>> import one_angle as oa
>> test = oa.rt_layers()
>> test.solve()
Similarly with the other models where one_angle is replaced with the other model module name.
There are various plotting and evaluation functions that can be used such as Scalar_flux() and plot_brdf(). See python help for descriptions eg.
>> ? oa.plot_brdf()
or
>> ? test.Scalar_flux()

A brief description of the files and modules:
radtran.py - This is the module that contains code used in other modules. Functions that are common to all RT codes are stored here. 
one_angel.py - This is the one angle RT code for homogeneous vegetation. This is the simplest code which does not include azimuth angles for the solution.
two_angle.py - This is the two angle RT code for homogeneous vegetation. This includes azimuths as well as sky downwelling radiation, Clumping, a hotspot, and specular reflection. 
ThreeD_angle.py - This is the 3D RT code. This is for heterogenous vegetation types and requires a file describing the vegetation.
p_theory.py - This is the module containing the formulas required in spectral invariant calculations. 
lad_model_quad.py - The program that creates the heterogenous vegetation structure file used by the 3D code. 
leaf_angle_distribution.py - This module contains the code for various leaf normal probability density function distributions. 
nose_test.py - The nosetest program that tests all the RT functions against known truth values. 
rami_test.py - A program that runs all the simulations for comparison to the RAMI-ROMC project. They are saved in the required format for upload to the ROMC website.   
