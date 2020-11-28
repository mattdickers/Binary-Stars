import numpy as np
import matplotlib.pyplot as plt
import Periodogram

name = 'lc_4987'

#period = 3.746684
#period = 2*1.8674005358922026
#period = 2*1.8736722091507432

phase = True
colour = False

#fileName = name + '_Coloured'
fileName = name

# Load Lightcurve (4987):
t, m, me, f1, f2 = np.genfromtxt('Data/'+name+'.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)

#lightcurve = np.array([t[np.where(t-2400000.5 < 53500)], m, me])
lightcurve = np.array([t, m, me])
period = Periodogram.periodogram(lightcurve, '4987', talk=0, plot=0)*2

# Load comparison lightcurves
t_c, m_c, me_c, f1_c, f2_c = np.genfromtxt('Data/lc_4860.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)
#t_c, m_c, me_c, f1_c, f2_c = np.genfromtxt('Data/lc_5089.csv', delimiter=',', usecols=(0,11,10,20,21), unpack=True)
#t_c, m_c, me_c, f1_c, f2_c = np.genfromtxt('Data/lc_5150.csv', delimiter=',', usecols=(0,11,10,20,21), unpack=True)

errorLim = np.nanmedian(me)  # Calculate error limit based on median of errors
errorLim_c = np.nanmedian(me_c)  # Calculate error limit based on median of errors

plt.cla()

t = t[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get time only when flags are equal to zero and errors within limit
m = m[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get mags only when flags are equal to zero and errors within limit
me = me[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get mag errors only when flags are equal to zero and errors within limit

t_c = t_c[np.where(
    (me_c < errorLim_c) & (f1_c < 4) & (f2_c < 4))]  # Get time only when flags are equal to zero and errors within limit
m_c = m_c[np.where(
    (me_c < errorLim_c) & (f1_c < 4) & (f2_c < 4))]  # Get mags only when flags are equal to zero and errors within limit
me_c = me_c[np.where(
    (me_c < errorLim_c) & (f1_c < 4) & (f2_c < 4))]  # Get mag errors only when flags are equal to zero and errors within limit

m_c = m_c[np.in1d(t_c, t)]  # Mask mags to only same times as main star
me_c = me_c[np.in1d(t_c, t)]  # Mask mags errors to only same times as main star
t_c = t_c[np.in1d(t_c, t)]  # Mask times to only same times as main star
m = m[np.in1d(t, t_c)]  # Mask mags to only same times as comparison star
me = me[np.in1d(t, t_c)]  # Mask mags to only same times as comparison star
t = t[np.in1d(t, t_c)]  # Mask times to only same times as comparison star

# t = t[np.where(t-2400000.5 > 53500)]
# m = m[np.where(t-2400000.5 > 53500)]
# me = me[np.where(t-2400000.5 > 53500)]
#
# t_c = t_c[np.where(t_c-2400000.5 > 53500)]
# m_c = m_c[np.where(t_c-2400000.5 > 53500)]
# me_c = me_c[np.where(t_c-2400000.5 > 53500)]

if colour:
    diffs = np.where(np.diff(t))[0] + 1
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(diffs))]
    for i in range(len(diffs)):
        if diffs[i-1] == diffs[len(diffs)-1]:
            split_t = t[0:diffs[i]]
            split_m = m[0:diffs[i]]
            if phase:
                plt.plot((split_t / period) % 1, split_m, ',', color = colors[i])
            else:
                plt.plot(split_t, split_m, ',', color=colors[i])
        else:
            split_t = t[diffs[i-1]:diffs[i]]
            split_m = m[diffs[i - 1]:diffs[i]]
            if phase:
                plt.plot((split_t / period) % 1, split_m, ',', color = colors[i])
            else:
                plt.plot(split_t, split_m, ',', color=colors[i])
else:
    if phase:
        plt.plot((t / period) % 1, m, 'k,')
        #plt.plot((t_c/period) % 1, m_c, 'r,')
    else:
        plt.plot(t, m, 'k,')
        #plt.plot(t_c, m_c, 'r,')

plt.gca().invert_yaxis()
plt.ylabel('Magnitude (mag)')

if phase:
    plt.xlabel('Phase')
    plt.savefig('lightcurves/'+fileName+('_Coloured' if colour else '')+'(Phase).png')
    plt.savefig('lightcurves/'+fileName+fileName+('_Coloured' if colour else '')+'(Phase).pdf')
else:
    plt.xlabel('Julian Date')
    plt.savefig('lightcurves/'+fileName+fileName+('_Coloured' if colour else '')+'(Day).png')
    plt.savefig('lightcurves/'+fileName+fileName+('_Coloured' if colour else '')+'(Day).pdf')
