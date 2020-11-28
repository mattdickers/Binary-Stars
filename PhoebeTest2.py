import phoebe as pb
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

#Define bundle for the system
b = pb.Bundle()
b.add_star('primary', mass=1.5, requiv=1.2) #Create star with mass 1.0 Msol
b.add_star('secondary', mass=1.0, requiv=1.0) #Create star with mass 1.5 Msol
b.add_orbit('binary', period=1.0) #Create binary systen with period 1.0 days
b.set_value('q', 1.5) #Define the mass ratio
b.set_hierarchy(pb.hierarchy.binaryorbit(b['binary'], b['primary'], b['secondary'])) #Create hierachy of the system

#Add datasets
b.add_dataset('lc', times = pb.linspace(0,10,20), passband='Johnson:R', dataset='lc01') #Add lc dataset with Johnson R passband
#b.add_dataset('orb', compute_times = pb.linspace(0,100,10), component=['primary', 'secondary'], dataset='orb01')

b.run_compute() #Forward compute properties

afig, mplfig = b['lc01'].plot(x='phase', save='Ligthcurve.png')
print('Plotted Lightcurve')

# b.set_value('incl@orbit', 80)
# b.run_compute(b='run_with_incl_80')
# afig, mplfig = b['orb01@run_with_incl_80'].plot(time=1.0, save='Orbit.png')
# print('Plotted Orbit')