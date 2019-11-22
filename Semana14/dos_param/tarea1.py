#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import pymultinest
import os
if not os.path.exists("chains"): os.mkdir("chains")

numpy.random.seed(42)
from colossus.cosmology import cosmology


pk_cmasdr12=numpy.loadtxt("GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt").T

cosmo = cosmology.setCosmology('planck15')
print(cosmo)


def Pk_Om(Om_,b, k, z):
    cosmo.Om0=Om_
    #print(Om_)
    #print(cosmo.Om0, b2)
    return (b**2)*cosmo.matterPowerSpectrum(k,z)


def chisq(theta, data):                 # Donde theta son los datos que se generan a aprtir del montecarlo 
    x= data[0]                          
    y= data[1]
    yerr= data[2]
    om=theta[0]
    b=theta[1]
    
    z=0.57
        
    model= Pk_Om(theta[0],theta[1], x, z)
    chisq= (y-model)**2 / yerr**2
    
    return chisq.sum()


# probability function, taken from the eggbox problem.
def myprior(cube):
    cube[0]=(0.05+0.9*cube[0])
    cube[1]=(0.1+5*cube[1])
    temp= cube
   # print(temp)
   # quit()
    return temp

def myloglike(cube):
    loglike=-0.5*chisq(cube,pk_cmasdr12) 
    return loglike

# number of dimensions our problem has
parameters = ["x", "y"]
n_params = len(parameters)
# name of the output files
prefix = "chains/3-"

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=prefix)
s = a.get_stats()
print(s)


# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, 
	n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
#print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)

with open('%sstats.json' %  a.outputfiles_basename, mode='w') as f:
  	json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

