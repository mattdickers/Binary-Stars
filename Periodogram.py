import numpy as np
import matplotlib.pyplot as plt
#from astropy.stats import LombScargle
from astropy.timeseries import LombScargle
from PyAstronomy.pyTiming import pyPDM
from scipy.optimize import curve_fit
import os
import time

def fitfunc_sin(time, p0, p1, p2, p3):
    z = p0 + p1 * np.sin( p2 + 2.0*np.pi*time/p3 )
    return z

def periodogram(lightcurve_in, name="", retparam="", fixperiod="", pdm=0, double=0, talk=0, plot=0, location=""):
    if name == "": name = 'test'
    if retparam == "": retparam = 'N'
    if talk == "": talk = 0
    minperiod = 0.01  # minimum period to check
    maxperiod = 2.0  # maximum period to check

    f = np.linspace(0.01, 10, 1000000)  # checks 10 to 100 days
    #f = np.linspace(0.26, 0.28, 1000000)  # checks 10 to 100 days
    lightcurve = []
    lightcurve.append(lightcurve_in[0][np.argsort(lightcurve_in[0])])
    lightcurve.append(lightcurve_in[1][np.argsort(lightcurve_in[0])])
    lightcurve.append(lightcurve_in[2][np.argsort(lightcurve_in[0])])
    lightcurve = np.array(lightcurve)

    if pdm == 0:
        pgram = LombScargle(lightcurve[0], lightcurve[1]).power(f)

        maxper = np.argmax(pgram[np.where(1.0 / f > minperiod)])
        period = fixperiod
    else:
        start = time.time()
        s = pyPDM.Scanner(minVal=0.01, maxVal=10, dVal=1e-5, mode="frequency")
        p = pyPDM.PyPDM(lightcurve[0], lightcurve[1])
        #f, theta = p.pdmEquiBinCover(10, 3, s)
        f, theta = p.pdmEquiBin(10, s)
        minTheta = np.argmin(theta)
        period = 1 / (f[minTheta])
        pgram = 1-theta
        fixperiod = 0
        end = time.time()
        print(end-start)

    if fixperiod == "":
        period = 1.0 * 1. / f[maxper]
    elif (float(fixperiod) < 0.0):
        fixperiod = float(fixperiod)
        check = np.where((1.0 / f > -0.9 * fixperiod) & (1.0 / f < -1.1 * fixperiod))
        f_cut = f[check[0]]
        pgram_cut = pgram[check[0]]
        maxper = np.argmax(pgram_cut)
        period = 1.0 / f_cut[maxper]
    try:
        fixperiod = -float(fixperiod)
    except ValueError:
        pass

    if (talk == 1):
        print("period: ", period)

    if plot == 1:
        # plot the periodogram
        plt.clf()
        plt.figure(figsize=(6, 4))
        plt.rcParams.update({'font.size': 8})
        rectangle = [0.1, 0.55, 0.35, 0.35]
        ax = plt.axes(rectangle)
        days = 1/f
        plt.plot(days[np.where(days <= (period*2 if not double else period*4))], pgram[np.where(days <= (period*2 if not double else period*4))],
                 'k-', label='  P=' + np.str(period)[:10] + 'days')
        #plt.xlim(0, 3.0 * period)
        # plt.xlim(0,6.0)
        plt.xlabel('Time (d)')
        plt.ylabel('frequency')
        plt.legend(loc='upper center', handletextpad=0.01)

        # plot the lightcurve
        rectangle = [0.55, 0.55, 0.35, 0.35]
        ax = plt.axes(rectangle)
        plt.plot(lightcurve[0] - 2400000.5, lightcurve[1], color='0.7', marker='.',
                 linestyle='solid', markersize=0.1)
        plt.plot(lightcurve[0] - 2400000.5, lightcurve[1], color='Red', marker='o',
                 linestyle='none', markersize=1)
        plt.ylim(np.median(lightcurve[1]) + 5.0 * np.std(lightcurve[1]),
                 np.median(lightcurve[1]) - 5.0 * np.std(lightcurve[1]))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=5)
        plt.xlabel('Time, MJD (d)')
        plt.ylabel('Magnitude {} (mag)'.format('R'))

        if not os.path.exists(location+name+'/'):
            os.mkdir(location+name+'/')

        plt.savefig(location+name+'/'+name+'_Periodogram.pdf', format='pdf', bbox_inches='tight', dpi=600)
        plt.savefig(location+name+'/'+name+'_Periodogram.png', format='png', bbox_inches='tight', dpi=600)
        plt.show()

    if double:
        return period*2
    else:
        return period