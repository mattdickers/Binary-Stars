import phoebe as pb
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt
import time

def timeConvert(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)

def Run(n, failCount):
    try:
        print('\nStarting', n, 'data points')
        start = time.time()
        # Define bundle for the system
        model = pb.Bundle()
        model.add_star('primary', mass=1.5, requiv=1.2, teff=7000)  # Create primary star
        model.add_star('secondary', mass=1.0, requiv=1.0, teff=6000)  # Create secondary star
        model.add_orbit('binary', period=1.0, sma=7)  # Create binary system orbit
        model.set_value('q', (model['value@mass@secondary@component'] / model[
            'value@mass@primary@component']))  # Define the mass ratio
        model.set_hierarchy(pb.hierarchy.binaryorbit(model['binary'], model['primary'],
                                                     model['secondary']))  # Create hierachy of the system
        # TODO look into better method of generating system using flip_constraint to define masses
        # TODO will either need to be defined after set_hierarchy, or use default bundle

        # Add lightcurve dataset
        # n=1005
        # n=100
        times = pb.linspace(0, 10, n)
        model.add_dataset('lc', times=times, passband='Johnson:R', dataset='lc01')

        # Forward Compute
        model.run_compute()

        # Add errors:
        C = 2 / 100  # Uncertainty as %
        fluxes = model.get_value('fluxes', context='b')
        sigmas = np.random.normal(0, C, size=times.shape)
        newFluxes = fluxes * (1 + sigmas)

        # Run b compute
        model = pb.default_binary()
        model.add_dataset('lc', times=times, fluxes=newFluxes, sigmas=np.full_like(newFluxes, fill_value=C))
        model.set_value('pblum_mode', 'dataset-scaled')
        model.plot(x='phase', legend=True, save='LigthcurveData.png', s=0.01, label='Data')
        print('Plotted Data')

        # Add EBAI Solver
        # b.add_solver('estimator.lc_geometry', solver='lcGeom_solver') #LC Geomtry Solver
        # b.add_solver('estimator.lc_periodogram', solver='lcPeriod_solver') #LC Periodogram Solver
        model.add_solver('estimator.ebai', solver='ebai_solver')  # Neural Network Solver

        # Run Solver
        model.run_solver(solver='ebai_solver', solution='ebai_solution')
        # solution = b.plot(solution='lc_solution', save='Solution.png')
        # print('Plotted Solution')
        # print(b.adopt_solution(trial_run=True))

        model.flip_constraint('requivsumfrac', solve_for='requiv@primary')
        model.flip_constraint('teffratio', solve_for='teff@primary')
        model.flip_constraint('esinw', solve_for='ecc')
        model.flip_constraint('ecosw', solve_for='per0')

        adopt_params = []
        for i, param in enumerate(model['value@fitted_values@ebai_solution']):
            if np.isnan(param):
                pass
            else:
                adopt_params.append(model['value@adopt_parameters@ebai_solution'][i])
        model['adopt_parameters@ebai_solution'] = adopt_params

        model.adopt_solution('ebai_solution')
        model.run_compute()
        model.plot(x='phase', ls='-', legend=True, save='EBAI_Solution.png', s=0.01, label='EBAI')
        print('Plotted EBAI Solution')

        # Add LC Geometry Solver
        model.add_solver('estimator.lc_geometry', solver='lcGeom_solver')  # LC Geomtry Solver

        model.run_solver('lcGeom_solver', solution='lcGeom_solution')

        model.flip_constraint('per0', solve_for='ecosw')
        model.flip_constraint('ecc', solve_for='esinw')

        model.adopt_solution('lcGeom_solution')
        model.run_compute()
        model.plot(x='phase', ls='-', legend=True, save='Geometry_Solution.png', s=0.01, label='Geometry')
        print('Plotted LC Geometry Solution')

        # #Add LC Periodogram Solver
        # b.add_solver('estimator.lc_periodogram', solver='lcPeriod_solver') #LC Periodogram Solver
        #
        # b.run_solver('lcPeriod_solver', solution='lcPeriod_solution')
        #
        # b.adopt_solution('lcPeriod_solution')
        # b.run_compute()
        # b.plot(x='phase', ls='-', legend=True, save='Periodogram_Solution.png', s=0.01, label='Periodogram')
        # print('Plotted LC Periodogram Solution')

        end = time.time()
        print('\nCompute Time:', timeConvert(end - start))

        compTimes.append(end - start)
        print('Complete\n')

    except (ValueError, AttributeError):
        failCount+=1
        print('FAIL\n')
        if failCount<100:
            Run(n, failCount)

global compTimes
compTimes = []

dataPoints = [100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000]
compTimes = [28.164461135864258, 116.04766273498535, 231.9618046283722, 367.631472826004, 452.8675866127014, 563.2299945354462,
             679.0984132289886, 792.0920975208282, 898.6561760902405, 1009.424154996872, 1124.2753274440765, 1234.547101020813,
             1346.411604642868, 1465.7015404701233, 1592.3056373596191, 1692.4058952331543, 1828.84628200531, 1981.7090103626251,
             2024.419947385788, 2135.462520122528, 2262.5722608566284]

# for i in dataPoints:
#     failCount = 0
#     Run(i, failCount)


times_seconds = np.array(compTimes)

times = np.array([timeConvert(time) for time in times_seconds])

fit = np.polyfit(times_seconds, dataPoints, 1)
fittedPoints = fit[0] * times_seconds + fit[1]

plt.cla()
plt.plot(dataPoints, times_seconds, 'k.')
plt.plot(fittedPoints, times_seconds, 'r-')
plt.yticks(np.array([0,  600, 1200, 1800, 2400]), (timeConvert(0), timeConvert(600), timeConvert(1200), timeConvert(1800), timeConvert(2400)))
plt.xlabel('Number of Datapoints')
plt.ylabel('Execution Time (minutes)')
plt.savefig('RunTimes.png')
print('Saved')