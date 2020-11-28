import phoebe as pb
from phoebe import u # units
import numpy as np
import time

logger = pb.logger(filename='PHOEBE.log')

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

start = time.time()

stars = {'Test': {'period': 5, 'm1': 1.5, 'm2': 1.0, 'requiv1': 1.3, 'requiv2': 1.0, 'teff1' : 6340, 'teff2' : 7630},
         'AQ Serpentis': {'period': 1.68743059, 'm1': 1.417, 'm2': 1.346, 'requiv1': 2.451, 'requiv2': 2.281, 'teff1': 6430.0, 'teff2': 6340.0},
         'M31': {'period': 3.549694, 'm1': 23.1, 'm2': 15.0, 'requiv1': 13.1, 'requiv2': 11.28},
         'ID 118': {'period': 2.760998, 'm1': 1.661, 'm2': 0.365, 'requiv1': 1.691, 'requiv2': 0.896, 'teff1': 7501.0, 'teff2': 6353.0},
         'ID 249': {'period': 2.319172, 'm1': 2.154, 'm2': 0.254, 'requiv1': 2.039, 'requiv2': 0.786, 'teff1': 5710.0, 'teff2': 5227.0}, #This one works at n=101
         'IM Persei': {'period': 2.25422694, 'm1': 1.7831, 'm2':  1.7741, 'requiv1': 2.409, 'requiv2': 2.366}, #Works with n=1005
         'V501 Herculis': {'period': 8.597687, 'm1': 1.269, 'm2': 1.211 , 'requiv1': 2.001, 'requiv2': 1.511, 'teff1': 5683.0, 'teff2': 5720.0}
         }

# Define bundle for the system
starName = 'IM Persei'
star = stars[starName]

b = pb.Bundle()

if 'teff1' and 'teff2' in list(star.keys()):
    b.add_star('primary', mass=star['m1'], requiv=star['requiv1'], teff=star['teff1'])  # Create primary star
    b.add_star('secondary', mass=star['m2'], requiv=star['requiv2'], teff=star['teff2'])  # Create secondary star
    b.add_orbit('binary', period=star['period'],
                sma=sma(star['period'], star['m1'], star['m2'])[2])  # Create binary system orbit
    b.set_value('q', star['m2'] / star['m1'])  # Define the mass ratio
else:
    b.add_star('primary', mass=star['m1'], requiv=star['requiv1'])  # Create primary star
    b.add_star('secondary', mass=star['m2'], requiv=star['requiv2'])  # Create secondary star
    b.add_orbit('binary', period=star['period'],
                sma=sma(star['period'], star['m1'], star['m2'])[2])  # Create binary system orbit
    b.set_value('q', star['m2'] / star['m1'])  # Define the mass ratio

b.set_hierarchy(pb.hierarchy.binaryorbit(b['binary'], b['primary'],
                                         b['secondary']))  # Create hierarchy of the system

#Generate Model Data
# b = pb.default_binary()
# # set parameter values
# b.set_value('q', value = 0.6)
# b.set_value('incl', component='binary', value = 84.5)
# b.set_value('ecc', 0.2)
# b.set_value('per0', 63.7)
# b.set_value('requiv', component='primary', value=1.)
# b.set_value('requiv', component='secondary', value=0.6)
# #b.set_value('teff', component='secondary', value=5500.)

# add an lc dataset
n = 1005
phases = pb.linspace(0, 1, n)
b.add_dataset('lc', compute_phases=phases, passband='Johnson:R')

#compute the b
b.run_compute(irrad_method='none')

# # Add errors:
# C = 2 / 100  # Uncertainty as %
# times = b.get_value('times', context='model', dataset='lc01')
# fluxes = b.get_value('fluxes', context='model', dataset='lc01')
# sigmas = np.random.normal(0, C, size=times.shape)
# newFluxes = fluxes * (1 + sigmas)
# #TODO the error calculation causes the crashes!

times = b.get_value('times', context='model', dataset='lc01')
np.random.seed(0)
newFluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
sigmas = np.ones_like(times) * 0.04

#Run Model Compute
b = pb.default_binary()
b.add_dataset('lc', times=times, fluxes=newFluxes, sigmas=sigmas, passband='Johnson:R')
b.set_value('period@binary', value=star['period'])
b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='Initial_Fit')
b.plot(x='phase', legend=True, s=0.01, save='lightcurves/Initial_Fit.png')
print('Initial Plotted\n')


#EBAI Solver
b.add_solver('estimator.ebai', solver='EBAI_solver')
#print(b['ebai01'])
b['phase_bin@EBAI_solver'] = False
#print(b['ebai01'])

b.run_solver(solver='EBAI_solver', solution='EBAI_solution')

b.flip_constraint('requivsumfrac', solve_for='requiv@secondary')

b.flip_constraint('teffratio', solve_for='teff@secondary')
b.flip_constraint('esinw', solve_for='ecc')
b.flip_constraint('ecosw', solve_for='per0')

adopt_params = []
for i, val in enumerate(b['value@fitted_values@EBAI_solution']):
    if np.isnan(val):
        pass
    else:
        adopt_params.append(b['value@adopt_parameters@EBAI_solution'][i])
b['adopt_parameters@EBAI_solution'] = adopt_params

b.adopt_solution('EBAI_solution')
b.run_compute(model='EBAI_Fit')
b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/EBAI_Fit.png')
print('EBAI Plotted\n')


#LC geometry Solver
b.add_solver('estimator.lc_geometry', solver='lcGeom_solver')
#print(b['lcgeom'])

b.run_solver(solver='lcGeom_solver', solution='lcGeom_solution')

b.flip_constraint('per0', solve_for='ecosw')
b.flip_constraint('ecc', solve_for='esinw')

b.adopt_solution('lcGeom_solution')
b.run_compute(model = 'LC_Geometry_Fit')
b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/LC_Geometry_Fit.png')
print('LC Geometry Plotted\n')

b.add_compute('ellc')
b.add_solver('optimizer.nelder_mead',
             fit_parameters=['teffratio', 'requivsumfrac', 'incl@binary', 'q', 'ecc', 'per0'])
b.run_solver(kind='nelder_mead', maxiter=10000, solution='nm_sol')
b.adopt_solution('nm_sol')
b.run_compute(model='after_nm')
b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/WithOptimizer.png')

end = time.time()
print('\nCompute Time:', timeConvert(end - start))
