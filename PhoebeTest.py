import phoebe as pb
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = pb.logger()
#Generate Binary Model
model = pb.default_binary()
model.add_dataset('lc', times=np.linspace(0,1,201))
#model.set_value(qualifier='period', component='binary', value=0.02)

#print(model['value@mass@primary@component'], model['value@mass@secondary@component'], model['value@period@orbit@component'])

#model.flip_constraint('mass@primary', solve_for='period')
#model.flip_constraint('mass@secondary', solve_for='q')

#model['mass@primary@component'] = 0.3
#model['mass@secondary@component'] = 0.3

#print(model['value@mass@primary@component'], model['value@mass@secondary@component'], model['value@period@orbit@component'])

#print(model['value@q@binary@component'])

model.run_compute()
afig, mplfig = model.plot(x='phase', save='Input.pdf')

#print('Primary Mass:',model.get_value('mass@primary@component')) #Primary Star Mass
#print('Secondary Mass:',model.get_value('mass@secondary@component')) #Secondary Star Mass

model['passband'] = 'Johnson:R' #Set passband to be Johnson R
print('Passband:',model.get_value('passband')) #Check which passband is used

#Compute moel based off input data (in this case the original model)
times = model.get_value('times', context='model')
fluxes = model.get_value('fluxes', context='model') + np.random.normal(size=times.shape) * 0.01
sigmas = np.ones_like(times) * 0.02

inverse = pb.default_binary()
inverse.add_dataset('lc', times=times, fluxes=fluxes, sigmas=np.full_like(fluxes, fill_value=0.1))
inverse.add_solver('estimator.lc_geometry', solver='my_lcgeom_solver')
#print(inverse.get_solver(solver='my_lcgeom_solver'))
inverse.run_solver(solver='my_lcgeom_solver', solution='my_lcgeom_solution')
sol = inverse.plot(solution='my_lcgeom_solution', save='Output.pdf')
print(inverse.adopt_solution())

#Investigate Constraints
#inverse.filter(context='constraint')
# print('Period:',inverse.get_value('period@orbit@component')) #Orbital Period
# print('Eccentricity:',inverse.get_value('ecc@orbit@component')) #Orbital Eccentricity
# print('Primary Inclination:',inverse.get_value('incl@primary@component')) #Primary Star Inclination
# print('Secondary Inclination:',inverse.get_value('incl@secondary@component')) #Secondary Star Inclination
# print('Primary Inclination:',inverse.get_value('mass@primary@component')) #Primary Star Mass
# print('Secondary Inclination:',inverse.get_value('mass@secondary@component')) #Secondary Star Mass
