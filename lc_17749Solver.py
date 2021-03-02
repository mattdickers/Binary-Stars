import phoebe as pb
from phoebe import u # units
import numpy as np
import time
import os

#logger = pb.logger(filename='PHOEBE.log')

def timeConvert(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)


#Create File Structure
fileName = 'lc_17749'
path = 'lightcurves/SolverFitted/'+fileName+'/'

#Create new folder for lightcurves
if not os.path.exists(path):
    os.mkdir(path)


start = time.time()

period, times, fluxes, sigmas = np.genfromtxt('lightcurves/Data/Running Median/' + fileName + '.csv', delimiter=',', unpack=True)
period = period[0]

b = pb.default_binary()
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R', dataset='lc01', overwrite=True)
b.set_value('period@binary', value=period)

# b.flip_constraint('mass@primary', 'sma@orbit')
# b.set_value('mass@primary@component', value=1.04)
#
# b.flip_constraint('mass@secondary', 'q')
# b.set_value('mass@secondary@component', value=0.8)

b.set_value('requiv@primary@star@component', value=1.07) #1.07

b.set_value('teff@primary@star@component', value=5800) #5800
b.set_value('teff@secondary@star@component', value=5500) #5200

b.set_value('incl@orbit', value=79) #80

b.set_value('t0_supconj', value=0.07)

b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='Initial_Fit')
b.plot(x='phase', legend=True, s=0.01, save=path+'Initial_Fit.png')
b.plot(x='phase', legend=True, s=0.01, save=path+'Initial_Fit.pdf')
print('Initial Plotted\n')


#LC Geometry Solver
print('Starting LC Goemetry Solver')
b.add_solver('estimator.lc_geometry', solver='lcGeom_solver')

b.run_solver(solver='lcGeom_solver', solution='lcGeom_solution')

print(b.adopt_solution('lcGeom_solution', trial_run=True))
b.adopt_solution('lcGeom_solution')
b.run_compute(model='LC_Geometry_Fit')
b.plot(x='phase', ls='-', legend=True, s=0.01, save=path+'LC_Geometry_Fit.png')
b.plot(x='phase', ls='-', legend=True, s=0.01, save=path+'LC_Geometry_Fit.pdf')
print('LC Geometry Plotted\n')


#Optimizer
print('Starting Optimizer Solver')
b.set_value_all('ld_mode', 'lookup')
b.add_compute('ellc', compute='fastcompute') #Add fastcompute option from ellc
b.add_solver('optimizer.nelder_mead',
             fit_parameters=['t0_supconj', 'incl@binary', 'q', 'ecc', 'per0'], compute='fastcompute')

b.run_solver(kind='nelder_mead', solution='optimizer_solution')

print(b.adopt_solution('optimizer_solution', trial_run=True))
b.adopt_solution('optimizer_solution')
b.run_compute(model='Optimizer_Fit', compute='fastcompute')
b.plot(x='phase', ls='-', legend=True, s=0.01, save=path+'WithOptimizer.png')
b.plot(x='phase', ls='-', legend=True, s=0.01, save=path+'WithOptimizer.pdf')
print('Optimizer Fit Plotted\n')