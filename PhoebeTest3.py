import phoebe as pb
from phoebe import u # units
import numpy as np

logger = pb.logger()

b = pb.default_binary()
# set parameter values
b.set_value('period', component='binary', value = 10.0)

# add an lc dataset
times = pb.linspace(0,1,101)
b.add_dataset('lc', compute_phases=times)

#compute the model
b.run_compute(irrad_method='none')

b.plot(x='phase', legend=True, save='1', s=0.01)

#Add errors:
C = 2/100 #Uncertainty as %
fluxes = b.get_value('fluxes', context='model')
sigmas = np.random.normal(0,C,size=times.shape)
newFluxes = fluxes * (1 + sigmas)

#Run model compute
b = pb.default_binary()
b.add_dataset('lc', times=times, fluxes=newFluxes, sigmas=np.full_like(newFluxes, fill_value=C))
b.set_value('pblum_mode', 'dataset-scaled')
b.plot(x='phase', legend=True, save='2.png', s=0.01, label='Data')

# # extract the arrays from the model that we'll use as observables in the next step
# times = b.get_value('times', context='model', dataset='lc01')
# # here we're adding noise to the fluxes as well to make the fake data more "realistic"
# np.random.seed(0) # to ensure reproducibility with added noise
# fluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
# sigmas_lc = np.ones_like(times) * 0.04
#
# b = phoebe.default_binary()
# b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas_lc)
# b.set_value('pblum_mode', 'dataset-scaled')
#
# b.run_compute(model='default')
# b.plot(x='phase', legend=True, save='1', s=0.01)
# print('Plotted 1')
