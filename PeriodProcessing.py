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


def magToFlux(mag, mag0):
    f = np.power(10, (mag0 - mag) / 2.5)
    #f /= max(f)
    f /= np.median(f)
    return f


def runningMedian(phase, mag):
    #rdelta = 0.005 / 2.0
    rdelta = 0.0005 / 0.5 #Increased phase space
    #rdelta = 0.005 / 0.2 #For 15284
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
        #print(rm[i],np.sort(mag[check[0]]),'\n')
        re[i] = np.std(mag[check[0]])
    return rp, rm, re


m0 = 4.93

name = 'lc_17749'

PDMperiods = {'lc_4987':3.74665143028274,
              'lc_15284':11.2227147747016,
              'lc_17749':2.47402276101259}

periodsMan = {'lc_4987':3.746690,
              'lc_15284':11.2227147747016,
              'lc_17749':2.47406}

limits = {'lc_4987':[0.49, 0.51, 11.3, 11.1],
          'lc_15284':[0.45, 0.55, 12.4, 12.0],
          'lc_17749':[0.51, 0.57, 12.25, 12.15]}

# Load Lightcurve:
t, m, me, f1, f2 = np.genfromtxt('Data/' + name + '.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)

lightcurve = np.array([t, m, me])
periodLS = Periodogram.periodogram(lightcurve, name, pdm=0, double=1, plot=1, location='lightcurves/Periodograms/', limits=(1.5,4))
periods = [periodLS, PDMperiods[name], periodsMan[name]]

errorLim = np.nanmedian(me)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 5))
plots = [ax1, ax2, ax3]
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

titles = {ax1:'Lomb-Scargle', ax2:'PDM', ax3:'Manual'}
labels = {ax1:dayConvert(periods[0]), ax2:dayConvert(periods[1]), ax3:dayConvert(periods[2])}

plt.cla()
plotValues = []
for i, period in enumerate(periods):
    t, m, me, f1, f2 = np.genfromtxt('Data/' + name + '.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)

    t = t[np.where(
        (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get time only when flags are equal to zero and errors within limit

    m = m[np.where(
        (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get mags only when flags are equal to zero and errors within limit
    me = me[np.where(
        (me < errorLim) & (f1 == 0) & (
                    f2 == 0))]  # Get mag errors only when flags are equal to zero and errors within limit

    badNights = np.where(((t < 2452980) | (t > 2453020))) #Select bad nights in centre
    t = t[badNights]
    m = m[badNights]
    me = me[badNights]

    m -= 4.93  # Convert instrumental mag to apparent


    #Calculate flux
    f = magToFlux(m, m0)

    # Calculate phase and running median
    ph = (t / period) % 1
    rp, rm, re = runningMedian(ph, m)
    rp, rf, rfe = runningMedian(ph, f)

    inliers = np.where(np.abs(rm - m) / re <= 3.0)  # Find points lying within sigma region
    t = t[inliers[0]]
    m = m[inliers[0]]
    me = me[inliers[0]]
    f1 = f1[inliers[0]]
    f2 = f2[inliers[0]]
    f = f[inliers[0]]

    # Recaculate running median
    ph = (t / period) % 1
    rp, rm, re = runningMedian(ph, m)
    rp, rf, rfe = runningMedian(ph, f)

    rpmin = rp[np.argmax(rm)]
    rp = (rp-rpmin) % 1
    ph = (ph-rpmin) % 1

    # Sort arrays by phase
    sort = np.argsort(ph)
    t = t[sort]
    ph = ph[sort]
    m = m[sort]
    me = me[sort]
    f1 = f1[sort]
    f2 = f2[sort]
    f = f[sort]

    sort = np.argsort(rp)
    rp = rp[sort]
    rm = rm[sort]
    re = re[sort]
    rf = rf[sort]
    rfe = rfe[sort]

    label = labels[plots[i]]

    plots[i].scatter(ph, m, color='black', s=1, label=label)
    plots[i].plot(rp, rm, color='blue', markersize=0.5)
    plots[i].fill_between(ph, rm-re, rm+re, color='cyan',
                     alpha=0.5)

    plots[i].invert_yaxis()
    # if plots[i] == ax1:
    #     plots[i].set_ylabel('Magnitude (mag)')
    # else:
    #     plots[i].get_yaxis().set_visible(False)
    #
    # if plots[i] == ax2:
    #     plots[i].set_xlabel('Phase')
    # else:
    #     pass

    plots[i].set_title(titles[plots[i]])
    plots[i].legend(loc='upper center')

    plots[i].set_xlim(limits[name][0], limits[name][1])
    plots[i].set_ylim(limits[name][2], limits[name][3])

plt.xlabel('Phase')
plt.ylabel('Magnitude (mag)', labelpad=22.0)
plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)

if not os.path.exists('lightcurves/Data/Periods/'+name+'/'):
    os.mkdir('lightcurves/Data/Periods/'+name+'/')

plt.savefig('lightcurves/Data/Periods/'+name+'/'+name+'Periods.png', bbox_inches='tight')
plt.savefig('lightcurves/Data/Periods/'+name+'/'+name+'Periods.pdf', bbox_inches='tight')
