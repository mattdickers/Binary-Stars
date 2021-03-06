import numpy as np
import matplotlib.pyplot as plt
import Periodogram
import os
import pandas as pd

def dayConvert(days):
    d = int(days)
    days = (days%1)*24
    h = int(days)
    days = (days%1)*60
    m = int(days)
    days = (days%1)*60
    s = int(days)

    return "%02dd:%02dh:%02dm:%02ds" % (d, h, m, s)


def runningMedian(phase, mag):
    rdelta = 0.005 / 0.5 #For 15284
    rp = phase
    rm = np.empty(len(rp))
    re = np.empty(len(rp))
    for i in range(len(rp)):
        check = np.where((phase < rp[i] + rdelta) & (phase > rp[i] - rdelta))
        if rp[i] - rdelta < 0:
            check = np.where((phase < rp[i] + rdelta) | (phase - 1 > rp[i] - rdelta))
        if rp[i] + rdelta > 1:
            check = np.where((phase > rp[i] - rdelta) | (phase + 1 < rp[i] + rdelta))
        rm[i] = np.median(mag[check[0]])
        re[i] = np.std(mag[check[0]])
    return rp, rm, re

# Load Lightcurve:
t, m, me = np.genfromtxt('Data/lc_15284ASAS-SN.csv', delimiter=',', unpack=True)

lightcurve = np.array([t, m, me])

periods = [11.2224577, 11.2224577/2]

for period in periods:
    plt.cla()

    ph = (t / period) % 1

    # Calculate phase and running median
    ph = (t / period) % 1
    rp, rm, re = runningMedian(ph, m)

    rpmin = rp[np.argmax(rm)]
    ph = (ph-rpmin) % 1
    rp = (rp-rpmin) % 1

    # Sort arrays by phase
    sort = np.argsort(ph)
    t = t[sort]
    ph = ph[sort]
    m = m[sort]
    me = me[sort]

    sort = np.argsort(rp)
    rp = rp[sort]
    rm = rm[sort]
    re = re[sort]

    fig, ax = plt.subplots()


    plt.scatter(ph, m, color='black', s=1, label='Period=%.7f days' % (period,))
    plt.plot(rp, rm, color='blue', markersize=0.5, label='Running Median')
    plt.fill_between(rp, rm-re, rm+re, color='cyan', alpha=0.5, label=r'$1\sigma$')

    plt.gca().invert_yaxis()
    plt.ylabel('Magnitude (mag)')

    plt.legend()

    plt.xlabel('Phase')
    plt.savefig('lightcurves/Data/lc_15284ASAS-SN/'+ ('Doubled' if period == 11.2224577/2 else 'Normal') + '.png')
    plt.savefig('lightcurves/Data/lc_15284ASAS-SN/'+ ('Doubled' if period == 11.2224577/2 else 'Normal') + '.pdf')