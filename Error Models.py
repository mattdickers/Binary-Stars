import phoebe as pb
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

#Define bundle for the system
model = pb.Bundle()
model.add_star('primary', mass=1.5, requiv=1.2) #Create star with mass 1.0 Msol
model.add_star('secondary', mass=1.0, requiv=1.0) #Create star with mass 1.5 Msol
model.add_orbit('binary', period=1.0) #Create binary systen with period 1.0 days
model.set_value('q', (model['value@mass@secondary@component'] / model['value@mass@primary@component'])) #Define the mass ratio
model.set_hierarchy(pb.hierarchy.binaryorbit(model['binary'], model['primary'], model['secondary'])) #Create hierachy of the system

#Add lightcurve dataset
times = pb.linspace(0,100,200)
model.add_dataset('lc', times = times, passband='Johnson:R', dataset='lc01')

#Compute
model.run_compute() #Forward compute properties

afig, mplfig = model['lc01'].plot(x='phase', save='Ligthcurve.png')
print('Plotted Lightcurve')

#Add errors:
C = 5/100 #
fluxes = model.get_value('fluxes', context='model')
sigmas = np.random.normal(0,C,size=times.shape)
newFluxes = fluxes * (1 + sigmas)

#Run inverse compute
inverse = pb.default_binary()
inverse.add_dataset('lc', times=times, fluxes=newFluxes, sigmas=np.full_like(newFluxes, fill_value=C))
inverse.add_solver('estimator.lc_geometry', solver='my_lcgeom_solver')
#print(inverse.get_solver(solver='my_lcgeom_solver'))
inverse.run_solver(solver='my_lcgeom_solver', solution='my_lcgeom_solution')
solution = inverse.plot(solution='my_lcgeom_solution', save='Solution.png')
print('Plotted Solution')
inverse.adopt_solution()

