import phoebe as pb
from phoebe import u # units
import autofig
import matplotlib.pyplot as plt
plt.rc('mathtext', fontset="cm")
import numpy as np
import time

fileName = 'IM_Persei'
path = 'lightcurves/SolverFitted/IM_Persei/'
optimisers = ['NM', 'Powell', 'CG']
optimiserNames = {'NM':'nelder_mead', 'Powell':'powell', 'CG':'cg'}


#Optimizer
for i in range(3):
    b = pb.Bundle.open(path + fileName + optimisers[i] + '.phoebe')

    b['lc01@dataset'].plot(x='phase', s=0.01, xlim=(-0.46,0.46), ylim=(1.825,2.06), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@Initial_Fit'].plot(x='phase', s=0.01, xlim=(-0.46,0.46), ylim=(1.825,2.06), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@EBAI_Fit'].plot(x='phase', ls='-', s=0.01, c='r', xlim=(-0.46,0.46), ylim=(1.825,2.06), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@LC_Geometry_Fit'].plot(x='phase', ls='-', s=0.01, c='g', xlim=(-0.46,0.46), ylim=(1.825,2.06), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@Optimizer_Fit'].plot(x='phase', ls='-', s=0.01, c='y', xlim=(-0.46,0.46), ylim=(1.825,2.06), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1,
                                 save=(path+'Optimisers/OptimierMaximas'+optimisers[i]+'.pdf'))
plt.cla()
for i in range(3):
    b = pb.Bundle.open(path + fileName + optimisers[i] + '.phoebe')

    b['lc01@dataset'].plot(x='phase', s=0.01, xlim=(-0.1,0.1), ylim=(0.9,1.25), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@Initial_Fit'].plot(x='phase', s=0.01, xlim=(-0.1,0.1), ylim=(0.9,1.25), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@EBAI_Fit'].plot(x='phase', ls='-', s=0.01, c='r', xlim=(-0.1,0.1), ylim=(0.9,1.25), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@LC_Geometry_Fit'].plot(x='phase', ls='-', s=0.01, c='g', xlim=(-0.1,0.1), ylim=(0.9,1.25), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1)
    b['lc01@Optimizer_Fit'].plot(x='phase', ls='-', s=0.01, c='y', xlim=(-0.1,0.1), ylim=(0.9,1.25), xlabel='Phase', ylabel='Flux', title=str(i), axorder=i+1,
                                 save=(path+'Optimisers/OptimierMainimas'+optimisers[i]+'.pdf'))