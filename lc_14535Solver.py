import phoebe as pb
from phoebe import u # units
import matplotlib.pyplot as plt
plt.rc('mathtext', fontset="cm")
import numpy as np
import time
import os

#logger = pb.logger(filename='PHOEBE.log')

def timeConvert(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)


#Create File Structure
fileName = 'lc_14535'
path = 'lightcurves/SolverFitted/'+fileName+'/'

#Create new folder for lightcurves
if not os.path.exists(path):
    os.mkdir(path)


start = time.time()

period, times, fluxes, sigmas = np.genfromtxt('lightcurves/Data/Running Median/' + fileName + '.csv', delimiter=',', unpack=True)
period = period[0]

b = pb.default_binary(contact_binary=True)
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R', dataset='lc01', overwrite=True)
b.set_value('period@binary', value=period)

b.flip_constraint('mass@primary', 'sma@orbit')
b.set_value('mass@primary@component', value=1.1) #1.1

b.flip_constraint('mass@secondary', 'q')
b.set_value('mass@secondary@component', value=1.0) #1.0

b.set_value('requiv@primary@star@component', value=1.21) #1.21

b.set_value('teff@primary@star@component', value=6300) #6300
b.set_value('teff@secondary@star@component', value=6000) #6000

b.set_value('incl@orbit', value=68) #68

b.set_value('t0_supconj', value=0.80) #0.80

b.set_value('pblum_mode', 'dataset-scaled')

#Add Other Datasets
b.add_dataset('orb', compute_times=pb.linspace(0,4,1001), dataset='orb01')
b.add_dataset('rv', compute_times=pb.linspace(0,1,1001), dataset='rv01')
b.add_dataset('mesh', compute_times=pb.linspace(0,1,51), dataset='mesh01')
b.set_value('columns', value=['teffs'])
#b.set_value('coordinates', value=['xyz'])

b.run_compute(irrad_method='none', model='Initial_Fit')
b['lc01'].plot(x='phase', legend=True, s=0.01, ylabel=r'Flux', xlabel='Phase', save=path+'Initial_Fit.png')
b['lc01'].plot(x='phase', legend=True, s=0.01, ylabel=r'Flux', xlabel='Phase', save=path+'Initial_Fit.pdf')
print('Initial Plotted\n')

#Plot Orbit
b['orb01@model'].plot(time=1.0, xlabel=r'$x$', ylabel=r'$y$', xunit='AU', yunit='AU', save=path+'Others/'+'Orbit.png')
b['orb01@model'].plot(time=1.0, xlabel=r'$x$', ylabel=r'$y$', xunit='AU', yunit='AU', save=path+'Others/'+'Orbit.pdf')
print('Orbit Plotted\n')

#Plot RV
b['rv01@model'].plot(x='phases', xlabel='Phase', ylable='Radial Velocity', save=path+'Others/'+'RV.png')
b['rv01@model'].plot(x='phases', xlabel='Phase', ylable='Radial Velocity', save=path+'Others/'+'RV.pdf')
print('RV Plotted\n')

#Plot Mesh
b['mesh01@model'].plot(time=0, xlabel=r'$x$', ylabel=r'$y$', fc='teffs', ec='none', draw_sidebars=True,
                       fclabel=r'$T_{eff}$', save=path+'Others/'+'Mesh.png')
b['mesh01@model'].plot(time=0, xlabel=r'$x$', ylabel=r'$y$', fc='teffs', ec='none', draw_sidebars=True,
                       fclabel=r'$T_{eff}$', save=path+'Others/'+'Mesh.pdf')
print('Mesh Plotted\n')

b['mesh01@model'].plot(times=pb.linspace(0,1,51), xlabel=r'$x$', ylabel=r'$y$', fc='teffs', ec='None', animate=True,
                       draw_sidebars=True, fclabel=r'$T_{eff}$', save='animations/'+fileName+'MeshAnimation.gif')
print('Mesh Animation Plotted\n')


end = time.time()
print('\nCompute Time:', timeConvert(end - start))