import numpy as np
import matplotlib.pyplot as plt
import Periodogram
import os
import pandas as pd


def magToFlux(mag, mag0):
    f = np.power(10, (mag0 - mag) / 2.5)
    #f /= max(f)
    f /= np.median(f)
    return f


def runningMedian(phase, mag):
    #rdelta = 0.005 / 2.0
    rdelta = 0.005 / 0.5 #Increased phase space
    rp = phase
    #rp = np.arange(0, 1, 0.001) #Note if phase is changes, so will rp, use numpy array. Currently gives 1000 data point TODO causes crash as the m array is still of default length
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
colour = False
plotComparison = False
flux = True
m0 = 4.93

realPeriods = {'lc_4987': 3.746684,
               'lc_3957': 4.8377,
               'lc_14535': 0.43549,
               'lc_16188': 0.43733,
               'lc_25679': 0.97934}

#name = 'lc_4987' #Default
name = 'lc_4987'
comparisons = [None, 'lc_4860', 'lc_5089', 'lc_5150']
comparison = comparisons[0]

# Load Lightcurve:
t, m, me, f1, f2 = np.genfromtxt('Data/' + name + '.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)

lightcurve = np.array([t, m, me])
period = Periodogram.periodogram(lightcurve, name, pdm=1, double=1, plot=1, location='lightcurves/Periodograms/')
print(period)


#period = realPeriods[name]
errorLim = np.nanmedian(me)  # Calculate error limit based on median of errors

# Load comparison lightcurve
if comparison:
    t_c, m_c, me_c, f1_c, f2_c = np.genfromtxt('Data/' + comparison + '.csv', delimiter=',',
                                               usecols=(0, 11, 12, 20, 21), unpack=True)
    errorLim_c = np.nanmedian(me_c)  # Calculate error limit based on median of errors

plt.cla()

t = t[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get time only when flags are equal to zero and errors within limit
m = m[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get mags only when flags are equal to zero and errors within limit
me = me[np.where(
    (me < errorLim) & (f1 == 0) & (
                f2 == 0))]  # Get mag errors only when flags are equal to zero and errors within limit

m -= 4.93  # Convert instrumental mag to apparent

if comparison:
    m_c -= 4.93  # Convert instrumental mag to apparent
    t_c = t_c[np.where(
        (me_c < errorLim_c) & (f1_c < 4) & (
                    f2_c < 4))]  # Get time only when flags are equal to zero and errors within limit
    m_c = m_c[np.where(
        (me_c < errorLim_c) & (f1_c < 4) & (
                    f2_c < 4))]  # Get mags only when flags are equal to zero and errors within limit
    me_c = me_c[np.where(
        (me_c < errorLim_c) & (f1_c < 4) & (
                    f2_c < 4))]  # Get mag errors only when flags are equal to zero and errors within limit

    m_c = m_c[np.in1d(t_c, t)]  # Mask mags to only same times as main star
    me_c = me_c[np.in1d(t_c, t)]  # Mask mags errors to only same times as main star
    t_c = t_c[np.in1d(t_c, t)]  # Mask times to only same times as main star
    m = m[np.in1d(t, t_c)]  # Mask mags to only same times as comparison star
    me = me[np.in1d(t, t_c)]  # Mask mags to only same times as comparison star
    t = t[np.in1d(t, t_c)]  # Mask times to only same times as comparison star

    ph_c = ph = (t_c / period) % 1  # Compariosn phase

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

# t = t[np.where(t-2400000.5 > 53500)]
# m = m[np.where(t-2400000.5 > 53500)]
# me = me[np.where(t-2400000.5 > 53500)]
#
# t_c = t_c[np.where(t_c-2400000.5 > 53500)]
# m_c = m_c[np.where(t_c-2400000.5 > 53500)]
# me_c = me_c[np.where(t_c-2400000.5 > 53500)]

fig, ax = plt.subplots()

if colour:
    diffs = np.where(np.diff(t))[0] + 1
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(diffs))]
    for i in range(len(diffs)):
        if diffs[i - 1] == diffs[len(diffs) - 1]:
            split_t = t[0:diffs[i]]
            split_m = m[0:diffs[i]]
            if phase:
                ax.plot((split_t / period) % 1, split_m, ',', color=colors[i])
            else:
                ax.plot(split_t, split_m, ',', color=colors[i])
        else:
            split_t = t[diffs[i - 1]:diffs[i]]
            split_m = m[diffs[i - 1]:diffs[i]]
            if phase:
                ax.plot((split_t / period) % 1, split_m, ',', color=colors[i])
            else:
                ax.plot(split_t, split_m, ',', color=colors[i])
else:
    if phase:
        plt.scatter(rp, (f if flux else m), color='black', s=1, label='Period=%.7f days' % (period,))
        plt.plot(rp, (rf if flux else rm), color='blue', markersize=0.5, label='Running Median')
        plt.fill_between(rp, (rf - rfe if flux else rm-re), (rf + rfe if flux else rm+re), color='cyan',
                         alpha=0.5, label=r'$1\sigma$')
    else:
        plt.scatter(t, (f if flux else m), color='black', s=1, label='Period=%.7f days' % (period,))

if flux:
    plt.ylabel(r'Flux $\left(\frac{\mathrm{W}}{\mathrm{m}^{2}}\right)$')
else:
    plt.gca().invert_yaxis()
    plt.ylabel('Magnitude (mag)')

plt.legend()

if not os.path.exists('lightcurves/Data/'+('Flux/' if flux else 'Mag/')+name+'/'):
    os.mkdir('lightcurves/Data/'+('Flux/' if flux else 'Mag/')+name+'/')

if phase:
    plt.xlabel('Phase')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Coloured' if colour else '') + ('_Flux' if flux else '') + '(Phase).png')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Coloured' if colour else '') + ('_Flux' if flux else '') + '(Phase).pdf')
else:
    plt.xlabel('Time, MJD (d)')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Coloured' if colour else '') + ('_Flux' if flux else '') + '(MJD).png')
    plt.savefig(
        'lightcurves/Data/' + ('Flux/' if flux else 'Mag/') + name + '/' + name + ('_Coloured' if colour else '') + ('_Flux' if flux else '') + '(MJD).pdf')

#Save Running Median Fux Data:
n = 1000 #The approx number of data points that the running median arrays are reduced to

t = t[0:len(t):int(len(t)/n)] #Reduce time array
rf = rf[0:len(rf):int(len(rf)/n)] #Reduce running median flux array
rfe = rfe[0:len(rfe):int(len(rfe)/n)] #Reduce running median flux error array
outRF = np.array([np.full(t.shape, period), t, rf, rfe]).transpose()
np.savetxt('lightcurves/Data/Running Median/' + name + '.csv', outRF, delimiter=',')
print('Saved Output')