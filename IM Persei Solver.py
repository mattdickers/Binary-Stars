import phoebe as pb
from phoebe import u # units
import numpy as np
import time

logger = pb.logger('info', filename='PHOEBE.log')

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

path = 'lightcurves/SolverFitted/IM_Persei/'

start = time.time()

#Create new bundle
b = pb.Bundle()

period = 2.25422694
m1 =  1.7831
m2 = 1.7741
r1 = 2.409
r2 = 2.366

b.add_star('primary', mass=m1, requiv=r1)
b.add_star('secondary', mass=m2, requiv=r2)
b.add_orbit('binary', period=period, sma=sma(period, m1, m2)[2])
b.set_value('q', m2/m1)

b.set_hierarchy(pb.hierarchy.binaryorbit(b['binary'], b['primary'], b['secondary']))


# Add LC dataset
n = 1005
phases = pb.linspace(0, 1, n)
b.add_dataset('lc', compute_phases=phases, passband='Johnson:R')

b.run_compute(irrad_method='none')

times = b.get_value('times', context='model', dataset='lc01')
np.random.seed(0)
fluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
sigmas = np.ones_like(times) * 0.04


#Run Model Compute
b = pb.default_binary()
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R')
b.set_value('period@binary', value=period)
b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='Initial_Fit')
b.plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'Initial_Fit.png')
b.plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'Initial_Fit.pdf')
print('Initial Plotted\n')


#EBAI Solver
print('Starting EBAI Solver')
b.add_solver('estimator.ebai', solver='EBAI_solver')
b['phase_bin@EBAI_solver'] = False

b.run_solver(solver='EBAI_solver', solution='EBAI_solution')

b.flip_constraint('requivsumfrac', solve_for='requiv@secondary')

b.flip_constraint('teffratio', solve_for='teff@secondary')
b.flip_constraint('esinw', solve_for='ecc')
b.flip_constraint('ecosw', solve_for='per0')

adopt_params = [b['value@adopt_parameters@EBAI_solution'][i] for i, param in enumerate(b['value@fitted_values@EBAI_solution']) if not np.isnan(param)]
b['adopt_parameters@EBAI_solution'] = adopt_params

print(b.adopt_solution('EBAI_solution', trial_run=True))
b.adopt_solution('EBAI_solution')
b.run_compute(model='EBAI_Fit')
b.plot(x='phase', ls='-', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'EBAI_Fit.png')
b.plot(x='phase', ls='-', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'EBAI_Fit.pdf')
print('EBAI Plotted\n')


#LC Geometry Solver
print('Starting LC Goemetry Solver')
b.add_solver('estimator.lc_geometry', solver='lcGeom_solver')

b.run_solver(solver='lcGeom_solver', solution='lcGeom_solution')

b.flip_constraint('per0', solve_for='ecosw')
b.flip_constraint('ecc', solve_for='esinw')

print(b.adopt_solution('lcGeom_solution', trial_run=True))
b.adopt_solution('lcGeom_solution')
b.run_compute(model='LC_Geometry_Fit')
b.plot(x='phase', ls='-', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'LC_Geometry_Fit.png')
b.plot(x='phase', ls='-', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'LC_Geometry_Fit.pdf')
print('LC Geometry Plotted\n')


#Optimizer
print('Starting Optimizer Solver')
b.set_value_all('ld_mode', 'lookup')
b.add_compute('ellc', compute='fastcompute') #Add fastcompute option from ellc
b.add_solver('optimizer.nelder_mead',
             fit_parameters=['teffratio@binary', 'requivsumfrac@binary', 'incl@binary', 'q', 'ecc', 'per0'], compute='fastcompute')

b.run_solver(kind='nelder_mead', solution='optimizer_solution')

print(b.adopt_solution('optimizer_solution', trial_run=True))
b.adopt_solution('optimizer_solution')
b.run_compute(model='Optimizer_Fit', compute='fastcompute')
b.plot(x='phase', ls='-', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'CGWithOptimizer.png')
b.plot(x='phase', ls='-', legend=True, s=0.01, xlabel='Phase', ylabel='Flux', save=path+'CGWithOptimizer.pdf')
print('Optimizer Fit Plotted\n')

b.save('bundle.phoebe')

#Sampler
b.add_solver('sampler.emcee', solver='emcee_solver')
b.set_value('compute', value='fastcompute', solver='emcee_solver')

b.set_value('pblum_mode', 'component-coupled') #Could also set to 'dataset-coupled'

b.progress_every_niters = 5

#Default distribution from example:
b.add_distribution({'sma@binary': pb.gaussian_around(0.1),
                    'incl@binary': pb.gaussian_around(5),
                    't0_supconj': pb.gaussian_around(0.001),
                    'requiv@primary': pb.gaussian_around(0.2),
                    'ecc@binary': pb.uniform(0, 0.005), #Set such that cannot be a negative value. Upper bound is approx estimarot value, 0.005
                    'sigmas_lnf@lc01': pb.uniform(-1e9, -1e4),
                   }, distribution='ball_around_guess')

# b.add_distribution({'teffratio': pb.gaussian_around(0.1),
#                     'requivsumfrac': pb.gaussian_around(5),
#                     't0_supconj': pb.gaussian_around(0.001),
#                     'ecc': pb.gaussian_around(0.2),
#                     'per0': pb.gaussian_around(0.2),
#                     'sigmas_lnf@lc01': pb.uniform(-1e9, -1e4),
#                    }, distribution='ball_around_guess')

b.run_compute(model='EMCEE_Fit', compute='fastcompute', sample_from='ball_around_guess',
              sample_num=20)

b.set_value('init_from', 'ball_around_guess')

b.set_value('nwalkers', solver='emcee_solver', value=12) #Define number of walkers. Must be twice number of parameters
b.set_value('niters', solver='emcee_solver', value=250) #Define number of iterations

b.run_solver('emcee_solver', solution='emcee_solution')

b.run_compute(model='EMCEE_Fit', compute='fastcompute', sample_from='emcee_solution',
              sample_num=20, overwrite=True)

b.adopt_solution('emcee_solution', distribution='emcee_posteriors')

#b.plot_distribution_collection(distribution='emcee_posteriors', save='lightcurves/SolverFitted/IM_Persei/Sampler.png')

print(b.uncertainties_from_distribution_collection(distribution='emcee_posteriors', sigma=3, tex=True))

end = time.time()
print('\nCompute Time:', timeConvert(end - start))