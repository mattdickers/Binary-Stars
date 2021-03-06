import phoebe as pb
from phoebe import u # units
import matplotlib.pyplot as plt
plt.rc('mathtext', fontset="cm")
import numpy as np
import time
import os

#logger = pb.logger(filename='PHOEBE.log')
pb.multiprocessing_off()

def timeConvert(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)

fullRun = True
sampler = True

#Create File Structure
fileName = 'lc_17749'
path = 'lightcurves/SolverFitted/'+fileName+'/'


if fullRun:
    file = open(path+fileName+'Outputs.txt', 'w')
    file.write('')

    #Create new folder for lightcurves
    if not os.path.exists(path):
        os.mkdir(path)


    start = time.time()

    period, times, fluxes, sigmas = np.genfromtxt('lightcurves/Data/Running Median/' + fileName + '.csv', delimiter=',', unpack=True)
    period = period[0]

    b = pb.default_binary()
    b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R', dataset='lc01', overwrite=True)
    b.set_value('period@binary', value=period)

    b.set_value('requiv@primary@star@component', value=1.07)

    b.set_value('teff@primary@star@component', value=5800)
    b.set_value('teff@secondary@star@component', value=5500)

    b.set_value('incl@orbit', value=79)

    b.set_value('t0_supconj', value=0.24)

    b.set_value('pblum_mode', 'dataset-scaled')

    b.run_compute(model='Initial_Fit')
    b['lc01@dataset'].plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux')
    #b.plot(x='phase', legend=True, s=0.01, save=path+'Initial_Fit.png')
    b['lc01@Initial_Fit'].plot(x='phase', legend=True, s=0.01,  xlabel='Phase', ylabel=r'Flux', save=path+'Initial_Fit.pdf')
    print('Initial Plotted\n')


    #LC Geometry Solver
    print('Starting LC Goemetry Solver')
    b.add_solver('estimator.lc_geometry', solver='lcGeom_solver')

    b.run_solver(solver='lcGeom_solver', solution='lcGeom_solution')

    print(b.adopt_solution('lcGeom_solution', trial_run=True))
    file.writelines('LC Geometry:\n'+str(b.adopt_solution('lcGeom_solution', trial_run=True))+'\n\n')
    b.adopt_solution('lcGeom_solution')
    b.run_compute(model='LC_Geometry_Fit')
    b['lc01@dataset'].plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux')
    b['lc01@Initial_Fit'].plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux')
    #b.plot(x='phase', ls='-', legend=True, s=0.01, save=path+'LC_Geometry_Fit.png')
    b['lc01@LC_Geometry_Fit'].plot(x='phase', ls='-', legend=True, s=0.01,  xlabel='Phase', ylabel=r'Flux', c='g', save=path+'LC_Geometry_Fit.pdf')
    print('LC Geometry Plotted\n')


    #Optimizer
    print('Starting Optimizer Solver')
    b.set_value_all('ld_mode', 'lookup')
    b.add_compute('ellc', compute='fastcompute') #Add fastcompute option from ellc
    b.add_solver('optimizer.nelder_mead',
                 fit_parameters=['t0_supconj', 'incl@binary', 'q', 'ecc', 'per0'], compute='fastcompute')

    b.run_solver(kind='nelder_mead', solution='optimizer_solution')

    print(b.adopt_solution('optimizer_solution', trial_run=True))
    file.writelines('Optimiser:\n'+str(b.adopt_solution('optimizer_solution', trial_run=True))+'\n\n')
    b.adopt_solution('optimizer_solution')
    b.run_compute(model='Optimizer_Fit', compute='fastcompute')
    b['lc01@dataset'].plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux')
    b['lc01@Initial_Fit'].plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux')
    b['lc01@LC_Geometry_Fit'].plot(x='phase', ls='-', legend=True, s=0.01, c='g', xlabel='Phase', ylabel=r'Flux')
    #b.plot(x='phase', ls='-', legend=True, s=0.01, save=path+'WithOptimizer.png')
    b['lc01@Optimizer_Fit'].plot(x='phase', ls='-', legend=True, s=0.01,  xlabel='Phase', ylabel=r'Flux', c='y', save=path+'WithOptimizer.pdf')
    print('Optimizer Fit Plotted\n')

    b.save(path+fileName+'.phoebe')
    print('Bundle Saved')

else:
    b = pb.Bundle.open(path+fileName+'.phoebe')

if sampler:
    #Sampler
    b.add_solver('sampler.emcee', solver='emcee_solver')
    b.set_value('compute', value='fastcompute', solver='emcee_solver')

    b.set_value('pblum_mode', 'component-coupled') #Could also set to 'dataset-coupled'

    #Default distribution from example:
    # b.add_distribution({'sma@binary': pb.gaussian_around(0.1),
    #                     'incl@binary': pb.gaussian_around(5),
    #                     't0_supconj': pb.gaussian_around(0.001),
    #                     'requiv@primary': pb.gaussian_around(0.2),
    #                     'pblum@primary': pb.gaussian_around(0.2),
    #                     'sigmas_lnf@lc01': pb.uniform(-1e9, -1e4),
    #                    }, distribution='ball_around_guess')

    b.add_distribution({'t0_supconj': pb.gaussian_around(0.001),
                        'incl@binary': pb.gaussian_around(5),
                        'q': pb.gaussian_around(1),
                        'ecc': pb.gaussian_around(0.005),
                        'per0': pb.gaussian_around(0.2),
                        'sigmas_lnf@lc01': pb.uniform(-1e9, -1e4),
                       }, distribution='ball_around_guess')

    b.run_compute(model='EMCEE_Fit', compute='fastcompute', sample_from='ball_around_guess',
                  sample_num=20)

    b.set_value('init_from', 'ball_around_guess')

    b.set_value('nwalkers', solver='emcee_solver', value=12) #Define number of walkers. Must be twice number of parameters
    b.set_value('niters', solver='emcee_solver', value=250) #Define number of iterations

    b.run_solver('emcee_solver', solution='emcee_solution')

    b.run_compute(model='EMCEE_Fit', compute='fastcompute', sample_from='emcee_solution',
                  sample_num=20, overwrite=True)

    b.adopt_solution('emcee_solution', distribution='emcee_posteriors')

    #b.plot_distribution_collection(distribution='emcee_posteriors', save='lightcurves/SolverFitted/Sampler.png')

    print(b.uncertainties_from_distribution_collection(distribution='emcee_posteriors', sigma=3, tex=True))
    file.writelines('Sampler:\n'+str(b.uncertainties_from_distribution_collection(distribution='emcee_posteriors', sigma=3,
                                                                                  tex=True))+'\n\n')
b['lc01@dataset'].plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux')
b['lc01@Initial_Fit'].plot(x='phase', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux')
b['lc01@EMCEE_Fit'].plot(x='phase', ls='-', legend=True, s=0.01, xlabel='Phase', ylabel=r'Flux', c='m', save=path+'Sampler.pdf')
file.close()


# #Add Other Datasets
# b.add_dataset('orb', compute_times=pb.linspace(0,4,1001), dataset='orb01')
# b.add_dataset('rv', compute_times=pb.linspace(0,4,1001), dataset='rv01')
# b.add_dataset('mesh', compute_times=pb.linspace(0,2,101), dataset='mesh01')
# b.set_value('columns', value=['teffs'])
# b.run_compute(compute='phoebe01')
#
# #Plot Orbit
# b['orb01@model'].plot(time=1.0, xlabel=r'$x$', ylabel=r'$y$', xunit='AU', yunit='AU', save=path+'Others/'+'Orbit.png')
# b['orb01@model'].plot(time=1.0, xlabel=r'$x$', ylabel=r'$y$', xunit='AU', yunit='AU', save=path+'Others/'+'Orbit.pdf')
# print('Orbit Plotted\n')
#
# #Plot RV
# b['rv01@model'].plot(x='phases', xlabel='Phase', ylable='Radial Velocity', save=path+'Others/'+'RV.png')
# b['rv01@model'].plot(x='phases', xlabel='Phase', ylable='Radial Velocity', save=path+'Others/'+'RV.pdf')
# print('RV Plotted\n')
#
# #Plot Mesh
# b['mesh01@model'].plot(time=1.0, xlabel=r'$x$', ylabel=r'$y$', fc='teffs', ec='none', fclim=(5000,6000), draw_sidebars=True,
#                        fclabel=r'$T_{eff}$', save=path+'Others/'+'Mesh.png')
# b['mesh01@model'].plot(time=0.5, xlabel=r'$x$', ylabel=r'$y$', fc='teffs', ec='none', fclim=(5000,6000), draw_sidebars=True,
#                        fclabel=r'$T_{eff}$', save=path+'Others/'+'Mesh.pdf')
# print('Mesh Plotted\n')
#
# b['mesh01@model'].plot(times=pb.linspace(0,2,101), xlabel=r'$x$', ylabel=r'$y$', fc='teffs', ec='None', fclim=(5000,6000), animate=True,
#                        draw_sidebars=True, fclabel=r'$T_{eff}$', save='animations/'+fileName+'MeshAnimation.gif')
# print('Mesh Animation Plotted\n')
#
# end = time.time()
# print('\nCompute Time:', timeConvert(end - start))