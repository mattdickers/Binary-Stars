import phoebe as pb
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt
import time

def timeConvert(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)

start = time.time()
#Define bundle for the system
# period = 3.549694
# model = pb.Bundle()
# model.add_star('primary', mass=23.1, requiv=13.1) #Create primary star
# model.add_star('secondary', mass=15.0, requiv=11.3) #Create secondary star
# model.add_orbit('binary', period=period, sma=33.0) #Create binary system orbit
# model.set_value('q', 0.65) #Define the mass ratio

# period = 1.68743059
# model = pb.Bundle()
# model.add_star('primary', mass=1.417, requiv=2.451, teff=6340.0) #Create primary star
# model.add_star('secondary', mass=1.346, requiv=2.281, teff=6430.0) #Create secondary star
# model.add_orbit('binary', period=period, sma=6.69957) #Create binary system orbit
# model.set_value('q', 0.94989) #Define the mass ratio

period = 1.0
model = pb.Bundle()
model.add_star('primary', mass=1.5, requiv=1.2) #Create primary star
model.add_star('secondary', mass=1.0, requiv=1.0) #Create secondary star
model.add_orbit('binary', period=period, sma=7) #Create binary system orbit
model.set_value('q', 1/1.5) #Define the mass ratio

model.set_hierarchy(pb.hierarchy.binaryorbit(model['binary'], model['primary'],
                                             model['secondary'])) #Create hierachy of the system
#TODO look into better method of generating system using flip_constraint to define masses
#TODO will either need to be defined after set_hierarchy, or use default bundle

#Add lightcurve dataset
n=1005
times = pb.linspace(0, 10, n)
model.add_dataset('lc', times=times, passband='Johnson:R', dataset='lc1')

#Forward Compute
model.run_compute()
model.plot(x='phase', legend=True, save='Original.png', s=0.01, label='Data')
print('Plotted Original')

#Add errors:
C = 2 / 100 #Uncertainty as %
fluxes = model.get_value('fluxes', context='model')
sigmas = np.random.normal(0, C, size=times.shape)
newFluxes = fluxes * (1 + sigmas)

#Run model compute
model = pb.default_binary()
model.set_value('period@binary', period)
model.add_dataset('lc', times=times, fluxes=newFluxes, sigmas=np.full_like(newFluxes, fill_value=C), passband='Johnson:R', dataset='lc2')
model.set_value('pblum_mode', 'dataset-scaled')
model.plot(x='phase', legend=True, save='LigthcurveData.png', s=0.01, label='Data')
print('Plotted Data')

#Add EBAI Solver
# _skip_filter_checks = {'check_default': False, 'check_visible': False}
# orbit = kwargs.get('orbit')
# global t0_supconj_param
# orbit_ps = model.get_component(component=orbit, **_skip_filter_checks)
# t0_supconj_param = orbit_ps.get_parameter(qualifier='t0_supconj', **_skip_filter_checks)  # Added line
model.add_solver('estimator.ebai', solver='ebai_solver') #Neural Network Solver

#Run Solver
model.run_solver(solver='ebai_solver', solution='ebai_solution')
#solution = model.plot(solution='lc_solution', save='Solution.png')
#print('Plotted Solution')
#print(model.adopt_solution(trial_run=True))

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
#
# #Add LC Geometry Solver
# model.add_solver('estimator.lc_geometry', solver='lcGeom_solver') #LC Geomtry Solver
#
# model.run_solver('lcGeom_solver', solution='lcGeom_solution')
#
# model.flip_constraint('per0', solve_for='ecosw')
# model.flip_constraint('ecc', solve_for='esinw')
#
# model.adopt_solution('lcGeom_solution')
# model.run_compute()
# model.plot(x='phase', ls='-', legend=True, save='Geometry_Solution.png', s=0.01, label='Geometry')
# print('Plotted LC Geometry Solution')
#
# # #Add LC Periodogram Solver
# # model.add_solver('estimator.lc_periodogram', solver='lcPeriod_solver') #LC Periodogram Solver
# #
# # model.run_solver('lcPeriod_solver', solution='lcPeriod_solution')
# #
# # model.adopt_solution('lcPeriod_solution')
# # model.run_compute()
# # model.plot(x='phase', ls='-', legend=True, save='Periodogram_Solution.png', s=0.01, label='Periodogram')
# # print('Plotted LC Periodogram Solution')
#
# end = time.time()
# print('\nCompute Time:', timeConvert(end - start))