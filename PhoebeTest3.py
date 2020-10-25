import phoebe
from phoebe import u # units
import numpy as np

logger = phoebe.logger()

b = phoebe.default_binary()
# set parameter values
b.set_value('q', value = 0.6)
#b.set_value('incl', component='binary', value = 84.5)
b.set_value('ecc', 0.2)
b.set_value('per0', 63.7)
b.set_value('requiv', component='primary', value=1.)
b.set_value('requiv', component='secondary', value=0.6)
b.set_value('teff', component='secondary', value=5500.)

# add an lc dataset
b.add_dataset('lc', compute_phases=phoebe.linspace(0,1,101))

#compute the model
b.run_compute(irrad_method='none')

# extract the arrays from the model that we'll use as observables in the next step
times = b.get_value('times', context='model', dataset='lc01')
# here we're adding noise to the fluxes as well to make the fake data more "realistic"
np.random.seed(0) # to ensure reproducibility with added noise
fluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
sigmas_lc = np.ones_like(times) * 0.04

b = phoebe.default_binary()
b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas_lc)
b.set_value('pblum_mode', 'dataset-scaled')

b.run_compute(model='default')
b.plot(x='phase', legend=True, save='1', s=0.01)
print('Plotted 1')
