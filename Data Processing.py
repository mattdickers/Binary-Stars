import numpy as np
import matplotlib.pyplot as plt
import Periodogram

phase = False
colour = False
plotComparison = False

name = 'lc_4987'
comparisons = [None, 'lc_4860', 'lc_5089', 'lc_5150']
comparison = comparisons[1]

#fileName = name + '_Coloured'
fileName = name

# Load Lightcurve:
t, m, me, f1, f2 = np.genfromtxt('Data/'+name+'.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)

#lightcurve = np.array([t[np.where(t-2400000.5 < 53500)], m, me])
lightcurve = np.array([t, m, me])
period = Periodogram.periodogram(lightcurve, name, plot=1, location='lightcurves/Periodograms/')*2
errorLim = np.nanmedian(me)  # Calculate error limit based on median of errors

# Load comparison lightcurve
if comparison:
    t_c, m_c, me_c, f1_c, f2_c = np.genfromtxt('Data/'+comparison+'.csv', delimiter=',', usecols=(0, 11, 12, 20, 21), unpack=True)
    errorLim_c = np.nanmedian(me_c)  # Calculate error limit based on median of errors

plt.cla()

t = t[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get time only when flags are equal to zero and errors within limit
m = m[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get mags only when flags are equal to zero and errors within limit
me = me[np.where(
    (me < errorLim) & (f1 == 0) & (f2 == 0))]  # Get mag errors only when flags are equal to zero and errors within limit

if comparison:
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

fig, ax = plt.subplots()

if colour:
    diffs = np.where(np.diff(t))[0] + 1
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i) for i in np.linspace(0, 1, len(diffs))]
    for i in range(len(diffs)):
        if diffs[i-1] == diffs[len(diffs)-1]:
            split_t = t[0:diffs[i]]
            split_m = m[0:diffs[i]]
            if phase:
                ax.plot((split_t / period) % 1, split_m, ',', color = colors[i])
            else:
                ax.plot(split_t, split_m, ',', color=colors[i])
        else:
            split_t = t[diffs[i-1]:diffs[i]]
            split_m = m[diffs[i - 1]:diffs[i]]
            if phase:
                ax.plot((split_t / period) % 1, split_m, ',', color = colors[i])
            else:
                ax.plot(split_t, split_m, ',', color=colors[i])
else:
    if phase:
        ax.plot((t / period) % 1, m, 'k,')
        if plotComparison:
            ax.plot((t_c/period) % 1, m_c, 'r,')
    else:
        ax.plot(t - 2400000.5, m, 'k,')
        if plotComparison:
            ax.plot(t_c, m_c, 'r,')

plt.gca().invert_yaxis()
plt.ylabel('Magnitude (mag)')


textstr = '\n'.join((
            r'$\mathrm{Period}=%.7f\mathrm{\,days}$' % (period,),
            r'$\mathrm{Comparison\,Star}=$'+(comparison if comparison else 'None')
            ))

props = dict(boxstyle='round', facecolor='white', alpha=0.5)

ax.text(0.05, 0.10, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

if phase:
    plt.xlabel('Phase')
    plt.savefig('lightcurves/Data/'+fileName+('_Coloured' if colour else '')+'(Phase).png')
    plt.savefig('lightcurves/Data/'+fileName+('_Coloured' if colour else '')+'(Phase).pdf')
else:
    plt.xlabel('Time, MJD (d)')
    plt.savefig('lightcurves/Data/'+fileName+('_Coloured' if colour else '')+'(Day).png')
    plt.savefig('lightcurves/Data/'+fileName+('_Coloured' if colour else '')+'(Day).pdf')
