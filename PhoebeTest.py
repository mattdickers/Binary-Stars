import phoebe as pb
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

# logger = phoebe.logger()
#
# b = phoebe.default_binary(contact_binary=True)
#
# b.add_dataset('mesh', compute_times=[0], dataset='mesh01')
# b.add_dataset('orb', compute_times=np.linspace(0,1,201), dataset='orb01')
# b.add_dataset('lc', times=np.linspace(0,1,21), dataset='lc01')
# b.add_dataset('rv', times=np.linspace(0,1,21), dataset='rv01')
#
# b.run_compute(irrad_method='none')
#
# print(b['mesh01@model'].components)
# afig, mplfig = b['mesh01@model'].plot(x='ws', save="PhoebeTest.png")
# afig, mplfig = b['mesh01@model'].plot(x='ws', save="PhoebeTest.pdf")

logger = pb.logger(clevel='WARNING')
#Generate Binary Model
model = pb.default_binary()
model.set_value(qualifier='teff', component='primary', value=6500)
model.add_dataset('lc', compute_times=pb.linspace(0,1,101))
model.run_compute()
afig, mplfig = model.plot(save='temp.png')

#Compute model based off input data (in this case the original model)
times = model.get_value('times', context='model')
fluxes = model.get_value('fluxes', context='model') + np.random.normal(size=times.shape) * 0.01
sigmas = np.ones_like(times) * 0.02

inverse = pb.default_binary()
inverse.add_dataset('lc', times=times, fluxes=fluxes, sigmas=np.full_like(fluxes, fill_value=0.1))
inverse.add_solver('estimator.lc_geometry', solver='my_lcgeom_solver')
#print(inverse.get_solver(solver='my_lcgeom_solver'))
inverse.run_solver(solver='my_lcgeom_solver', solution='my_lcgeom_solution')
sol = inverse.plot(solution='my_lcgeom_solution', save='temp.png')
print(inverse.adopt_solution())

#Investigate Constraints
#inverse.filter(context='constraint')
print('Period:',inverse.get_value('period@orbit@component')) #Orbital Peirod
print('Eccentricity:',inverse.get_value('ecc@orbit@component')) #Orbital Eccentricity
print('Primary Inclination:',inverse.get_value('incl@primary@component')) #Primary Star Inclination
print('Secondary Inclination:',inverse.get_value('incl@secondary@component')) #Secondary Star Inclination
print('Primary Inclination:',inverse.get_value('mass@primary@component')) #Primary Star Mass
print('Secondary Inclination:',inverse.get_value('mass@secondary@component')) #Secondary Star Mass
