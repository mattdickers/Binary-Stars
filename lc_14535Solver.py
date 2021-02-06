import phoebe as pb
from phoebe import u # units
import numpy as np
import time
import os

#logger = pb.logger(filename='PHOEBE.log')

def timeConvert(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)

def sma(P, m1, m2):
    G = 6.67430e-11  # Gravitational Constant
    SolarMass = 1.989e+30  # Solar Mass in kg
    day = 60 * 60 * 24  # Day in seconds

    # Convert to SI units:
    P = P * day  # Days to seconds
    m1 = m1 * SolarMass  # Solar Masses to kg
    m2 = m2 * SolarMass  # Solar Masses to kg

    # Calculate SMA in meters
    a = np.cbrt(((P ** 2) * G * (m1 + m2)) / (4 * (np.pi ** 2)))

    aAU = a / 1.496e+11  # Convert SMA to Solar Radii
    aR = a / 6.957e+8  # Convert SMA to AU
    return [a, aAU, aR]


#Create File Structure
fileName = 'lc_14535'
path = 'lightcurves/SolverFitted/'+fileName+'/'

#Create new folder for lightcurves
if not os.path.exists(path):
    os.mkdir(path)

start = time.time()

period, times, fluxes, sigmas = np.genfromtxt('lightcurves/Data/Running Median/' + fileName + '.csv', delimiter=',', unpack=True)
period = period[0]

b = pb.default_binary(contact_binary=True)
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R', dataset='lc01', overwrite=True)
b.set_value('period@binary', value=period)

b.flip_constraint('mass@primary', 'sma@orbit')
b.set_value('mass@primary@component', value=1.1)

b.flip_constraint('mass@secondary', 'q')
b.set_value('mass@secondary@component', value=1.0)

b.set_value('requiv@primary@star@component', value=1.21)

b.set_value('teff@primary@star@component', value=6300)
b.set_value('teff@secondary@star@component', value=6000)

b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='Initial_Fit')
b.plot(x='phase', legend=True, s=0.01, save=path+'Initial_Fit.png')
b.plot(x='phase', legend=True, s=0.01, save=path+'Initial_Fit.pdf')
print('Initial Plotted\n')

# b = pb.Bundle()
# b.add_star('primary', mass=1.07, requiv=1.18, teff=6000)
# b.add_star('secondary', mass=0.95, requiv=1.00, teff=6800)
# b.add_orbit('binary', period=period, sma=sma(period, 1.04, 1.00)[2])
# b.add_envelope('contact_envelope')
# b.set_value('q', 1.00 / 1.04)
#
# b.set_hierarchy(pb.hierarchy.binaryorbit(b['binary'], b['primary'], b['secondary'], b['contact_envelope']))
#
# b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R', dataset='lc01', overwrite=True)
#
# b.set_value('pblum_mode', 'dataset-scaled')
#
# b.run_compute(model='Initial_Fit')
# b.plot(x='phase', legend=True, s=0.01, save=path+'Initial_Fit.png')
# b.plot(x='phase', legend=True, s=0.01, save=path+'Initial_Fit.pdf')
# print('Initial Plotted\n')


#EBAI Solver
print('Starting EBAI Solver')
b.add_solver('estimator.ebai', solver='EBAI_solver')
b['phase_bin@EBAI_solver'] = False

b.run_solver(solver='EBAI_solver', solution='EBAI_solution')

#b.flip_constraint('requivsumfrac', solve_for='requiv@secondary')

b.flip_constraint('teffratio', solve_for='teff@secondary')
b.flip_constraint('esinw', solve_for='ecc')
b.flip_constraint('ecosw', solve_for='per0')

adopt_params = [b['value@adopt_parameters@EBAI_solution'][i] for i, param in enumerate(b['value@fitted_values@EBAI_solution']) if not np.isnan(param)]
b['adopt_parameters@EBAI_solution'] = adopt_params

print(b.adopt_solution('EBAI_solution', trial_run=True))
b.adopt_solution('EBAI_solution')
b.run_compute(model='EBAI_Fit')
b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/EBAI_Fit.png')
b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/EBAI_Fit.pdf')
print('EBAI Plotted\n')