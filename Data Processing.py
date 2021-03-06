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
    #rdelta = 0.005 / 0.5 #Increased phase space
    rdelta = 0.005 / 0.2 #For 15284
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


phase = True
flux = False
PDM = False
overridePeriod = True
m0 = 4.93

manualPeriods = {'lc_4987':3.746690,
                 'lc_14535':0.435490993254457,
                 'lc_15284':11.2227147747016/2,
                 'lc_17749':2.47406}

PDMperiods = {'lc_4987':3.74665143028274,
              'lc_14535':0.435490993254457,
              'lc_15284':11.2227147747016,
              'lc_17749':2.47402276101259}

#name = 'lc_4987' #Default
name = 'lc_15284'

# Load Lightcurve:
t, m, me, f1, f2 = np.genfromtxt('Data/' + name + '.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)

lightcurve = np.array([t, m, me])

if overridePeriod:
    period = manualPeriods[name]
elif PDM:
    period = PDMperiods[name]
else:
    period = Periodogram.periodogram(lightcurve, name, pdm=0, double=1, plot=1, location='lightcurves/Periodograms/')
print(period)
print(dayConvert(period))


errorLim = 2*np.nanmedian(me)  # Calculate error limit based on median of errors #For 15284
#errorLim = np.nanmedian(me)

plt.cla()

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

fig, ax = plt.subplots()


if phase:
    plt.scatter(rp, (f if flux else m), color='black', s=1, label='Period=%.7f days' % (period,))
    plt.plot(rp, (rf if flux else rm), color='blue', markersize=0.5, label='Running Median')
    plt.fill_between(rp, (rf - rfe if flux else rm-re), (rf + rfe if flux else rm+re), color='cyan',
                     alpha=0.5, label=r'$1\sigma$')
else:
    plt.scatter(t, (f if flux else m), color='black', s=1, label='Period=%.7f days' % (period,))

if flux:
    plt.ylabel(r'Flux $\left[\frac{\mathrm{W}}{\mathrm{m}^{2}}\right]$')
else:
    plt.gca().invert_yaxis()
    plt.ylabel('Magnitude (mag)')

plt.legend()

if not os.path.exists('lightcurves/Data/'+('Flux/' if flux else 'Mag/')+name+'/'):
    os.mkdir('lightcurves/Data/'+('Flux/' if flux else 'Mag/')+name+'/')

if phase:
    plt.xlabel('Phase')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Flux' if flux else '') + ('(PDM)' if PDM else ('(LS)' if not overridePeriod else '')) + '.png')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Flux' if flux else '') + ('(PDM)' if PDM else ('(LS)' if not overridePeriod else '')) + '.pdf')
else:
    plt.xlabel('Time, MJD (d)')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Flux' if flux else '') + '(MJD).png')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Flux' if flux else '') + '(MJD).pdf')

#Save Running Median Fux Data:
n = 1000 #The approx number of data points that the running median arrays are reduced to

# t = t[0:len(t):int(len(t)/n)] #Reduce time array
# rf = rf[0:len(rf):int(len(rf)/n)] #Reduce running median flux array
# rfe = rfe[0:len(rfe):int(len(rfe)/n)] #Reduce running median flux error array
# outRF = np.array([np.full(t.shape, period), t, rf, rfe]).transpose()
# np.savetxt('lightcurves/Data/Running Median/' + name + '.csv', outRF, delimiter=',')
# print('Saved Output')