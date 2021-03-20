import phoebe as pb
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.interpolate import make_interp_spline
import time

def timeConvert(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)

def sma(P, m1, m2):
    G = 6.67430e-11  # Gravitational Constant
    SolarMass = 1.989e+30  # Solar Mass in kg
    day = 60 * 60 * 24  # Day in seconds

    # Convert to SI units:
    P = P * day  # Days to seconds
    m1 = m1 * SolarMass  # Solar Masses to kg
    m2 = m2 * SolarMass  # Solar Masses to kg

    # Calculate SMA in meters
    a = np.cbrt(((P ** 2) * G * (m1 + m2)) / (4 * (np.pi ** 2)))

    aAU = a / 1.496e+11  # Convert SMA to Solar Radii
    aR = a / 6.957e+8  # Convert SMA to AU
    return [a, aAU, aR]

path = 'lightcurves/SolverFitted/IM_Persei/Efficiency/'
file = open(path+'Times.txt', 'w')
file.write('')

period = 2.25422694
m1 = 1.7831
m2 = 1.7741
r1 = 2.409
r2 = 2.366

dataPoints = [100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]
#dataPoints = [101,1000,2001,3001,4001,5001,6001,7001,8001,9001,10001]
#compTimes = [606.1696403026581, 1473.6362385749817, 968.9245252609253, 1559.3514025211334, 3901.952570915222, 2583.7632653713226, 1802.1300160884857, 2886.0005428791046, 2071.4019186496735, 2124.1891026496887, 2412.690982103348]
compTimes = []

# for i in dataPoints:
#     print('\nStarting', i, 'data points')
#     start = time.time()
#     b = pb.Bundle()
#
#     b.add_star('primary', mass=m1, requiv=r1)
#     b.add_star('secondary', mass=m2, requiv=r2)
#     b.add_orbit('binary', period=period, sma=sma(period, m1, m2)[2])
#     b.set_value('q', m2 / m1)
#
#     b.set_hierarchy(pb.hierarchy.binaryorbit(b['binary'], b['primary'], b['secondary']))
#
#     phases = pb.linspace(0, 1, i)
#     b.add_dataset('lc', compute_phases=phases, passband='Johnson:R')
#
#     b.run_compute(irrad_method='none')
#
#     times = b.get_value('times', context='model', dataset='lc01')
#     np.random.seed(0)
#     fluxes = b.get_value('fluxes', context='model', dataset='lc01') + np.random.normal(size=times.shape) * 0.02
#     sigmas = np.ones_like(times) * 0.04
#
#     # Run Model Compute
#     b = pb.default_binary()
#     b.add_dataset('lc', times=times, fluxes=fluxes, sigmas=sigmas, passband='Johnson:R')
#     b.set_value('period@binary', value=period)
#     b.set_value('pblum_mode', 'dataset-scaled')
#
#     b.run_compute(model='Initial_Fit')
#     print('Initial Done\n')
#
#     # EBAI Solver
#     print('Starting EBAI Solver')
#     b.add_solver('estimator.ebai', solver='EBAI_solver')
#     b['phase_bin@EBAI_solver'] = False
#
#     b.run_solver(solver='EBAI_solver', solution='EBAI_solution')
#
#     b.flip_constraint('requivsumfrac', solve_for='requiv@secondary')
#
#     b.flip_constraint('teffratio', solve_for='teff@secondary')
#     b.flip_constraint('esinw', solve_for='ecc')
#     b.flip_constraint('ecosw', solve_for='per0')
#
#     adopt_params = [b['value@adopt_parameters@EBAI_solution'][i] for i, param in
#                     enumerate(b['value@fitted_values@EBAI_solution']) if not np.isnan(param)]
#     b['adopt_parameters@EBAI_solution'] = adopt_params
#
#     #print(b.adopt_solution('EBAI_solution', trial_run=True))
#     b.adopt_solution('EBAI_solution')
#     b.run_compute(model='EBAI_Fit')
#     print('EBAI Done\n')
#
#     # LC Geometry Solver
#     print('Starting LC Goemetry Solver')
#     b.add_solver('estimator.lc_geometry', solver='lcGeom_solver')
#
#     b.run_solver(solver='lcGeom_solver', solution='lcGeom_solution')
#
#     b.flip_constraint('per0', solve_for='ecosw')
#     b.flip_constraint('ecc', solve_for='esinw')
#
#     #print(b.adopt_solution('lcGeom_solution', trial_run=True))
#     b.adopt_solution('lcGeom_solution')
#     b.run_compute(model='LC_Geometry_Fit')
#     print('LC Geometry Done\n')
#
#     # Optimizer
#     print('Starting Optimizer Solver')
#     b.set_value_all('ld_mode', 'lookup')
#     b.add_compute('ellc', compute='fastcompute')  # Add fastcompute option from ellc
#     b.add_solver('optimizer.nelder_mead',
#                  fit_parameters=['teffratio@binary', 'requivsumfrac@binary', 't0_supconj', 'incl@binary', 'q', 'ecc',
#                                  'per0'],
#                  compute='fastcompute')
#
#     b.run_solver(kind='nelder_mead', solution='optimizer_solution')
#
#     #print(b.adopt_solution('optimizer_solution', trial_run=True))
#     b.adopt_solution('optimizer_solution')
#     b.run_compute(model='Optimizer_Fit', compute='fastcompute')
#     print('Optimizer Done\n')
#
#     b.save(path + str(i) + '.phoebe')
#     print('Bundle Saved')
#
#     end = time.time()
#     compTime = end-start
#     compTimes.append(compTime)
#     file.writelines(str(i)+': '+str(compTime)+'\n')
# file.close()



#compTimes = [561.1358201503754, 883.8353290557861, 3536.118881225586, 1796.7914276123047, 2063.4513506889343, 2155.846523284912, 6398.320641994476, 2982.2705006599426, 3571.6684770584106, 5331.8255960941315, 7171.550770521164]
compTimes = [28.164461135864258, 116.04766273498535, 231.9618046283722, 367.631472826004, 452.8675866127014, 563.2299945354462,
             679.0984132289886, 792.0920975208282, 898.6561760902405, 1009.424154996872, 1124.2753274440765, 1234.547101020813,
             1346.411604642868, 1465.7015404701233, 1592.3056373596191, 1692.4058952331543, 1828.84628200531, 1981.7090103626251,
             2024.419947385788, 2135.462520122528, 2262.5722608566284]
times_seconds = np.array(compTimes)

times_seconds2 = np.array([561.1358201503754, 883.8353290557861, 1796.7914276123047, 2063.4513506889343, 2155.846523284912, 2982.2705006599426, 3571.6684770584106, 5331.8255960941315, 7171.550770521164])
dataPoints2 = [101,1000,3001,4001,5001,7001,8001,9001,10001]

def func(xs, a, b):
    return a*b**xs

# popt, pcov = optimize.curve_fit(func,dataPoints2,times_seconds2)
# fittedPoints = func(dataPoints2, popt[0], popt[1])
# print('y=$'+str(popt[0]*popt[1])+'^x')
# x = np.linspace(0,10200,300)
# spline = make_interp_spline(dataPoints2, fittedPoints)
# y = spline(x)


fit = np.polyfit(times_seconds, dataPoints, 1)
fittedPoints = fit[0] * times_seconds + fit[1]
print('y='+str(fit[0])+'x+'+str(fit[1]))

plt.cla()
plt.plot(dataPoints, times_seconds, 'k.', label='Data')
plt.plot(fittedPoints, times_seconds, 'r-', label='Fitted Function')
#plt.plot(x, y, 'r-', label='Fitted Function')
plt.yticks(np.array([0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600, 7200]), (timeConvert(0),
                    timeConvert(600), timeConvert(1200), timeConvert(1800), timeConvert(2400), timeConvert(3000), timeConvert(3600),
                    timeConvert(4200), timeConvert(4800), timeConvert(5400), timeConvert(6000), timeConvert(6600), timeConvert(7200)))
#plt.yticks(np.array([0,  600, 1200, 1800, 2400]), (timeConvert(0), timeConvert(600), timeConvert(1200), timeConvert(1800), timeConvert(2400)))
plt.xlabel('Number of Datapoints')
plt.ylabel('Execution Time (minutes)')
plt.legend()
plt.savefig(path+'RunTimes.pdf')
#plt.savefig(path+'OptimiserRunTimes.pdf')
print('Saved')