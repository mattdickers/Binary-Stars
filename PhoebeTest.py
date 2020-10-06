import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary(contact_binary=True)

b.add_dataset('mesh', compute_times=[0], dataset='mesh01')
b.add_dataset('orb', compute_times=np.linspace(0,1,201), dataset='orb01')
b.add_dataset('lc', times=np.linspace(0,1,21), dataset='lc01')
b.add_dataset('rv', times=np.linspace(0,1,21), dataset='rv01')

b.run_compute(irrad_method='none')

print(b['mesh01@model'].components)
afig, mplfig = b['mesh01@model'].plot(x='ws', save="PhoebeTest.png")