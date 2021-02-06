import phoebe as pb
from phoebe import u # units
import numpy as np

b = pb.Bundle()

period = 3.549694

b.add_star('primary', mass=23.1, requiv=13.1)
b.add_star('secondary', mass=15.0, requiv=11.28)
b.add_orbit('binary', period=period, sma=32.95462)
b.set_value('q', 15.0/23.1)

b.set_hierarchy(pb.hierarchy.binaryorbit(b['binary'], b['primary'],
                                         b['secondary']))

# Add LC dataset
n = 1005
phases = pb.linspace(0, 1, n)
b.add_dataset('lc', compute_phases=phases, passband='Johnson:R')

b.run_compute(irrad_method='none')

times = b.get_value('times', context='model', dataset='lc01')
np.random.seed(0)
fluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
sigmas = np.ones_like(times) * 0.04

b = pb.default_binary()
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R')
b.set_value('period@binary', value=period)
b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='Initial_Fit')


# #EBAI Solver
# b.add_solver('estimator.ebai', solver='EBAI_solver')
# b['phase_bin@EBAI_solver'] = False
#
# b.run_solver(solver='EBAI_solver', solution='EBAI_solution')
#
# b.flip_constraint('requivsumfrac', solve_for='requiv@secondary')
#
# b.flip_constraint('teffratio', solve_for='teff@secondary')
# b.flip_constraint('esinw', solve_for='ecc')
# b.flip_constraint('ecosw', solve_for='per0')
#
# adopt_params = [b['value@adopt_parameters@EBAI_solution'][i] for i, param in enumerate(b['value@fitted_values@EBAI_solution']) if not np.isnan(param)]
# b['adopt_parameters@EBAI_solution'] = adopt_params
#
# b.adopt_solution('EBAI_solution')
# b.run_compute(model='EBAI_Fit')


#LC Geometry Solver
b.add_solver('estimator.lc_geometry', solver='lcGeom_solver')

b.run_solver(solver='lcGeom_solver', solution='lcGeom_solution')

#b.flip_constraint('per0', solve_for='ecosw')
#b.flip_constraint('ecc', solve_for='esinw')

b.adopt_solution('lcGeom_solution')
b.run_compute(model = 'LC_Geometry_Fit')