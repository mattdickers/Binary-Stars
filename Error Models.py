import phoebe as pb
from phoebe import u  # units
import numpy as np
import matplotlib.pyplot as plt
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


# Custom run_compute function that can handle atmosphere errors
def run_compute(model):
    try:
        model.run_compute()
    except ValueError as error:
        if "Could not compute ldint with ldatm='ck2004'" in str(error):
            print('Cannot compute atmosphere. Using blackbody approximation.')

            model['atm@primary'] = 'blackbody'
            model['ld_mode@primary'] = 'manual'
            model['ld_func@primary'] = 'logarithmic'

            model['atm@secondary'] = 'blackbody'
            model['ld_mode@secondary'] = 'manual'
            model['ld_func@secondary'] = 'logarithmic'

            model.run_compute()


start = time.time()

# Test Stars
stars = {'Test': {'period': 5, 'm1': 1.5, 'm2': 1.0, 'requiv1': 1.3, 'requiv2': 1.0, 'teff1' : 6340, 'teff2' : 7630},
         'AQ Serpentis': {'period': 1.68743059, 'm1': 1.417, 'm2': 1.346, 'requiv1': 2.451, 'requiv2': 2.281, 'teff1': 6430.0, 'teff2': 6340.0},
         'M31': {'period': 3.549694, 'm1': 23.1, 'm2': 15.0, 'requiv1': 13.1, 'requiv2': 11.28},
         'ID 118': {'period': 2.760998, 'm1': 1.661, 'm2': 0.365, 'requiv1': 1.691, 'requiv2': 0.896, 'teff1': 7501.0, 'teff2': 6353.0},
         'ID 249': {'period': 2.319172, 'm1': 2.154, 'm2': 0.254, 'requiv1': 2.039, 'requiv2': 0.786, 'teff1': 5710.0, 'teff2': 5227.0},
         'IM Persei': {'period': 2.25422694, 'm1': 1.7831, 'm2':  1.7741, 'requiv1': 2.409, 'requiv2': 2.366},
         'V501 Herculis': {'period': 8.597687, 'm1': 1.269, 'm2': 1.211 , 'requiv1': 2.001, 'requiv2': 1.511, 'teff1': 5683.0, 'teff2': 5720.0}
         }

# Define bundle for the system
starName = 'ID 118'
star = stars[starName]

model = pb.Bundle()

if 'teff1' and 'teff2' in list(star.keys()):
    model.add_star('primary', mass=star['m1'], requiv=star['requiv1'], teff=star['teff1'])  # Create primary star
    model.add_star('secondary', mass=star['m2'], requiv=star['requiv2'], teff=star['teff2'])  # Create secondary star
    model.add_orbit('binary', period=star['period'],
                    sma=sma(star['period'], star['m1'], star['m2'])[2])  # Create binary system orbit
    model.set_value('q', star['m2'] / star['m1'])  # Define the mass ratio
else:
    model.add_star('primary', mass=star['m1'], requiv=star['requiv1'])  # Create primary star
    model.add_star('secondary', mass=star['m2'], requiv=star['requiv2'])  # Create secondary star
    model.add_orbit('binary', period=star['period'],
                    sma=sma(star['period'], star['m1'], star['m2'])[2])  # Create binary system orbit
    model.set_value('q', star['m2'] / star['m1'])  # Define the mass ratio

model.set_hierarchy(pb.hierarchy.binaryorbit(model['binary'], model['primary'],
                                             model['secondary']))  # Create hierarchy of the system

# Add lightcurve dataset
n = 1005
times = pb.linspace(0, 10, n)
model.add_dataset('lc', times=times, passband='Johnson:R', dataset='lc1')

# Forward Compute
model.run_compute()
# run_compute(b)

model.plot(x='phase', legend=True, save='LigthcurveData.png', s=0.01, label='Data')
print('Plotted Original')

# Add errors:
C = 2 / 100  # Uncertainty as %
fluxes = model.get_value('fluxes', context='b')
sigmas = np.random.normal(0, C, size=times.shape)
newFluxes = fluxes * (1 + sigmas)

# Run b compute
#b = pb.default_binary()
model = pb.Bundle()

if 'teff1' and 'teff2' in list(star.keys()):
    model.add_star('primary', mass=star['m1'], requiv=star['requiv1'], teff=star['teff1'])  # Create primary star
    model.add_star('secondary', mass=star['m2'], requiv=star['requiv2'], teff=star['teff2'])  # Create secondary star
    model.add_orbit('binary', period=star['period'],
                    sma=sma(star['period'], star['m1'], star['m2'])[2])  # Create binary system orbit
    model.set_value('q', star['m2'] / star['m1'])  # Define the mass ratio
else:
    model.add_star('primary', mass=star['m1'], requiv=star['requiv1'])  # Create primary star
    model.add_star('secondary', mass=star['m2'], requiv=star['requiv2'])  # Create secondary star
    model.add_orbit('binary', period=star['period'],
                    sma=sma(star['period'], star['m1'], star['m2'])[2])  # Create binary system orbit
    model.set_value('q', star['m2'] / star['m1'])  # Define the mass ratio

model.set_hierarchy(pb.hierarchy.binaryorbit(model['binary'], model['primary'],
                                             model['secondary']))  # Create hierarchy of the system

#b.set_value('period@binary', value=star['period'])
model.add_dataset('lc', times=times, fluxes=newFluxes, sigmas=np.full_like(newFluxes, fill_value=C), passband='Johnson:R', dataset='lc2')
model.set_value('pblum_mode', 'dataset-scaled')
model.plot(x='phase', legend=True, save='LigthcurveData.png', s=0.01, label='Data')
print('Plotted LigthcurveData')

# Add EBAI Solver
model.add_solver('estimator.ebai', solver='ebai_solver')  # Neural Network Solver

# Run Solver
model.run_solver(solver='ebai_solver', solution='ebai_solution')
# solution = b.plot(solution='lc_solution', save='Solution.png')
# print('Plotted Solution')
# print(b.adopt_solution(trial_run=True))

model.flip_constraint('requivsumfrac', solve_for='requiv@primary')
model.flip_constraint('teffratio', solve_for='teff@primary')
model.flip_constraint('esinw', solve_for='ecc')
model.flip_constraint('ecosw', solve_for='per0')

adopt_params = []
for i, param in enumerate(model['value@fitted_values@ebai_solution']):
    if np.isnan(param):
        pass
    else:
        adopt_params.append(model['value@adopt_parameters@ebai_solution'][i])
model['adopt_parameters@ebai_solution'] = adopt_params

model.adopt_solution('ebai_solution')
model.run_compute()
model.plot(x='phase', ls='-', legend=True, save='EBAI_Solution.png', s=0.01, label='EBAI')
print('Plotted EBAI Solution')

# Add LC Geometry Solver
model.add_solver('estimator.lc_geometry', solver='lcGeom_solver')  # LC Geomtry Solver

model.run_solver('lcGeom_solver', solution='lcGeom_solution')

model.flip_constraint('per0', solve_for='ecosw')
model.flip_constraint('ecc', solve_for='esinw')

model.adopt_solution('lcGeom_solution')
model.run_compute()
model.plot(x='phase', ls='-', legend=True, save='Geometry_Solution.png', s=0.01, label='Geometry')
print('Plotted LC Geometry Solution')

end = time.time()
print('\nCompute Time:', timeConvert(end - start))
