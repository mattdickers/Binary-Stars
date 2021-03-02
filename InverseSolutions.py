import phoebe as pb
from phoebe import u # units
import numpy as np
import time

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

def run_compute(b, model=None, irrad_method=None):
    try:
        if model and irrad_method:
            b.run_compute(model=model, irrad_method=irrad_method)
        elif model:
            b.run_compute(model=model)
        elif irrad_method:
            b.run_compute(irrad_method=irrad_method)
        else:
            b.run_compute()
    except ValueError as error:
        print(error)
        if "Could not compute ldint with ldatm='ck2004'" in str(error):
            print('Cannot compute atmosphere. Using blackbody approximation.')

            b['atm@primary'] = 'blackbody'
            b['ld_mode@primary'] = 'manual'
            b['ld_func@primary'] = 'logarithmic'

            b['atm@secondary'] = 'blackbody'
            b['ld_mode@secondary'] = 'manual'
            b['ld_func@secondary'] = 'logarithmic'

        elif "Could not lookup ld_coeffs for ld_coeffs_source_bol@primary@star@component" in str(error):
            print('Cannot determine bolometric coefficients. Disabling.')

            #b['ld_mode_bol@primary'] = 'manual'
            #b['ld_coeffs_source_bol'] = 0
            irrad_method = 'none'

        if model and irrad_method:
            run_compute(b, model=model, irrad_method=irrad_method)
        elif model:
            run_compute(b, model=model)
        elif irrad_method:
            run_compute(b, irrad_method=irrad_method)
        else:
            run_compute(b)

realData = True

#Enable Certain Parts of Code
EBAI = False
LCgeom = True

start = time.time()

#lc_14535 needs lc_bol fix
#lc_16188 needs lc_bol fix

if realData:
    fileName = 'lc_16188'
    period, times, fluxes, sigmas = np.genfromtxt('lightcurves/Data/Running Median/' + fileName + '.csv', delimiter=',', unpack=True)
    period = period[0]
else:
    stars = {'Test': {'period': 5, 'm1': 1.5, 'm2': 1.0, 'requiv1': 1.3, 'requiv2': 1.0, 'teff1' : 6340, 'teff2' : 7630},
             'AQ Serpentis': {'period': 1.68743059, 'm1': 1.417, 'm2': 1.346, 'requiv1': 2.451, 'requiv2': 2.281, 'teff1': 6430.0, 'teff2': 6340.0},
             'M31': {'period': 3.549694, 'm1': 23.1, 'm2': 15.0, 'requiv1': 13.1, 'requiv2': 11.28},
             'ID 118': {'period': 2.760998, 'm1': 1.661, 'm2': 0.365, 'requiv1': 1.691, 'requiv2': 0.896, 'teff1': 7501.0, 'teff2': 6353.0},
             'ID 249': {'period': 2.319172, 'm1': 2.154, 'm2': 0.254, 'requiv1': 2.039, 'requiv2': 0.786, 'teff1': 5710.0, 'teff2': 5227.0},
             'IM Persei': {'period': 2.25422694, 'm1': 1.7831, 'm2':  1.7741, 'requiv1': 2.409, 'requiv2': 2.366}, #Works with n=1005
             'V501 Herculis': {'period': 8.597687, 'm1': 1.269, 'm2': 1.211 , 'requiv1': 2.001, 'requiv2': 1.511, 'teff1': 5683.0, 'teff2': 5720.0}
             }

    # Define bundle for the system
    starName = 'M31'
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

    #Add LC dataset
    n = 1005 #Set the number of datapoints to use
    phases = pb.linspace(0, 1, n)
    b.add_dataset('lc', compute_phases=phases, passband='Johnson:R')

    b.run_compute(irrad_method='none')

    # # Add errors:
    # C = 2 / 100  # Uncertainty as %
    # times = b.get_value('times', context='model', dataset='lc01')
    # fluxes = b.get_value('fluxes', context='model', dataset='lc01')
    # sigmas = np.random.normal(0, C, size=times.shape)
    # newFluxes = fluxes * (1 + sigmas)
    # #TODO This method causes crashes, look into a new method. Using example for now

    times = b.get_value('times', context='model', dataset='lc01')
    np.random.seed(0)
    fluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
    sigmas = np.ones_like(times) * 0.04

#Run Model Compute
b = pb.default_binary()
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R')
b.set_value('period@binary', value=(period if realData else star['period']))
b.set_value('pblum_mode', 'dataset-scaled')

run_compute(b, model='Initial_Fit')
b.plot(x='phase', legend=True, s=0.01, save='lightcurves/SolverFitted/Initial_Fit.png')
b.plot(x='phase', legend=True, s=0.01, save='lightcurves/SolverFitted/Initial_Fit.pdf')
print('Initial Plotted\n')


#EBAI Solver
if EBAI:
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
    b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/EBAI_Fit.png')
    b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/EBAI_Fit.pdf')
    print('EBAI Plotted\n')


#LC Geometry Solver
if LCgeom:
    print('Starting LC Goemetry Solver')
    b.add_solver('estimator.lc_geometry', solver='lcGeom_solver')

    b.run_solver(solver='lcGeom_solver', solution='lcGeom_solution')

    b.flip_constraint('per0', solve_for='ecosw')
    b.flip_constraint('ecc', solve_for='esinw')

    print(b.adopt_solution('lcGeom_solution', trial_run=True))
    b.adopt_solution('lcGeom_solution')
    b.run_compute(model = 'LC_Geometry_Fit')
    b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/LC_Geometry_Fit.png')
    b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/LC_Geometry_Fit.pdf')
    print('LC Geometry Plotted\n')


#Optimizer
print('Starting Optimizer Solver')
b.set_value_all('ld_mode', 'lookup')
b.add_compute('ellc', compute='fastcompute') #Add fastcompute option from ellc
b.add_solver('optimizer.nelder_mead',
             fit_parameters=['teffratio@binary', 'requivsumfrac@binary', 'incl@binary', 'q', 'ecc', 'per0'], compute='fastcompute')

b.run_solver(kind='nelder_mead', solution='optimizer_solution')

b.adopt_solution('optimizer_solution')
b.run_compute(model='Optimizer_Fit', compute='fastcompute')
b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/WithOptimizer.png')
b.plot(x='phase', ls='-', legend=True, s=0.01, save='lightcurves/SolverFitted/WithOptimizer.pdf')
print('Optimizer Fit Plotted\n')


#Sampler
b.add_solver('sampler.emcee', solver='emcee_solver')
b.set_value('compute', value='fastcompute', solver='emcee_solver')

b.set_value('pblum_mode', 'component-coupled') #Could also set to 'dataset-coupled'

#Default distribution from example:
b.add_distribution({'sma@binary': pb.gaussian_around(0.1),
                    'incl@binary': pb.gaussian_around(5),
                    't0_supconj': pb.gaussian_around(0.001),
                    'requiv@primary': pb.gaussian_around(0.2),
                    'pblum@primary': pb.gaussian_around(0.2),
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

b.plot_distribution_collection(distribution='emcee_posteriors', save='lightcurves/SolverFitted/Sampler.png')

print(b.uncertainties_from_distribution_collection(distribution='emcee_posteriors', sigma=3, tex=True))

end = time.time()
print('\nCompute Time:', timeConvert(end - start))
