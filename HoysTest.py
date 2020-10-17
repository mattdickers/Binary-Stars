import numpy as np
import matplotlib.pyplot as plt
import all_hc_functions_indented as hc

lightcurve = np.load('WeiredIC1396Aobject.npy')

#hc.make_plot_lightcurve(lightcurve)
#hc.probplot(lightcurve)
#lc_temp = hc.extract_data(lightcurve,prop_name='name',invert='')
hc.periodogram(lightcurve)