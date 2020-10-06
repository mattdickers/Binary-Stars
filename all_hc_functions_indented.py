# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:23:15 2020

"""

from __future__ import print_function
from astropy.coordinates import match_coordinates_sky , SkyCoord
from astropy import stats
from astropy import units as u
from astropy.stats import LombScargle
from matplotlib.font_manager import FontProperties
from operator import itemgetter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import optimize #Leastsq Levenberg-Marquadt Algorithm
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
import argparse
import math
import matplotlib 
import matplotlib.pylab as plt
import matplotlib.pyplot as  pyplot
import numpy
import os
import pandas as pd
import pprint
import pymysql.cursors
import scipy.spatial.distance as ssd
import matplotlib.colors as colors
from astropy.time import Time
##################################
#MODIFIED VERSION OF QUERY SCRIPT#
##################################



class ServerTalkException(Exception):
    pass

class NoStarsException(Exception):
    pass


USER = 'sps_student'
PASSWORD = 'sps_student'
HA_FILTERS = ['HA', 'A', 'A-BAND', 'HA-BAND', 'H-BAND', 'H BAND', 'HA BAND', 'H-ALPHA', 'H ALPHA', 'HALPHA']


def index_stars(coords, radius, dec, connection):
    """
    Get the lightcurve data for a set of coordinates
    :param coords:
    :param radius:
    :param dec:
    :param form:
    :return:
    """
    print('Retriving data from server...')
    with connection.cursor() as cursor:
        # Get all stars within radius
        sql = "SELECT * FROM photometry WHERE alpha_j2000 BETWEEN %s-(%s/3600 / COS(%s * PI() / 180)) AND %s+(%s/3600 / COS(%s * PI() / 180)) AND delta_j2000 BETWEEN %s-%s/3600 AND %s+%s/3600;"

        cursor.execute(sql, (coords.ra.degree, radius, dec, coords.ra.degree, radius, dec, dec, radius, dec, radius))

        stars = cursor.fetchall()
    print('...data retrived!')

    print('Indexing stars...')

    lightcurve_data = {}
    lightcurve_data['stars'] = []
    lightcurve_data['filters'] = []

    for star in stars:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM observations WHERE id = %s"
            cursor.execute(sql, (star['observation_id']))
            observation = cursor.fetchone()
        if observation['orignal_filter'].upper() in HA_FILTERS:
            lightcurve_data['stars'].append({'date': observation['date'],
                                             'calibrated_magnitude': star['calibrated_magnitude'],
                                             'alpha_j2000': float(star['alpha_j2000']),
                                             'delta_j2000': float(star['delta_j2000']),
                                             'calibrated_error': star['calibrated_error'],
                                             'id': star['id'],
                                             'filter': 'HA',
                                             'original_filter': observation['orignal_filter'],
                                             'x': star['x'],
                                             'y': star['y'],
                                             'magnitude_rms_error': star['magnitude_rms_error'],
                                             'fwhm_world': star['fwhm_world'],
                                             'observation_id': observation['id'],
                                             'user_id': observation['user_id'],
                                             'target': observation['target_id'],
                                             'flags': star['flags'],
                                             'magnitude': star['magnitude'],
                                             'device_id': observation['device_id'],
                                             'fits_id': observation['fits_id']})
            lightcurve_data['filters'].append('HA')
        else:
            lightcurve_data['stars'].append({'date': observation['date'],
                                             'calibrated_magnitude': star['calibrated_magnitude'],
                                             'alpha_j2000': float(star['alpha_j2000']),
                                             'delta_j2000': float(star['delta_j2000']),
                                             'calibrated_error': star['calibrated_error'],
                                             'id': star['id'],
                                             'filter': observation['filter'],
                                             'original_filter': observation['orignal_filter'],
                                             'x': star['x'],
                                             'y': star['y'],
                                             'magnitude_rms_error': star['magnitude_rms_error'],
                                             'fwhm_world': star['fwhm_world'],
                                             'observation_id': observation['id'],
                                             'user_id': observation['user_id'],
                                             'target': observation['target_id'],
                                             'flags': star['flags'],
                                             'magnitude': star['magnitude'],
                                             'device_id': observation['device_id'],
                                             'fits_id': observation['fits_id']})
            lightcurve_data['filters'].append(observation['filter'])

    # No stars for the coords? Then we can't continue
    if len(lightcurve_data['stars']) == 0:
        raise NoStarsException("No stars for input co-ordinates")

    # Below here in this function is exactly the same as the website
    # Get rid of duplicates
    lightcurve_data['filters'] = list(set(lightcurve_data['filters']))

    # Build up some lists of stars, co-ordinates and magnitudes
    stars_for_filter = sorted(lightcurve_data['stars'], key=itemgetter('calibrated_magnitude'), reverse=True)
    coord_list = numpy.array(
        [numpy.array(list(map(itemgetter('alpha_j2000'), stars_for_filter))), numpy.array(list(map(itemgetter('delta_j2000'), stars_for_filter)))])
    mag_list = numpy.array(list(map(itemgetter('calibrated_magnitude'), stars_for_filter)))

    sep = distance_matrix(numpy.transpose(coord_list), numpy.transpose(coord_list))

    square = ssd.squareform(sep * 3600.)

    linkage_of_square = linkage(square, 'single')

    clusters = fcluster(linkage_of_square, 3, criterion='distance')

    median_ra = numpy.zeros(int(numpy.max(clusters)), dtype=float)
    median_dec = numpy.zeros(int(numpy.max(clusters)), dtype=float)
    median_mag = numpy.zeros(int(numpy.max(clusters)), dtype=float)

    lightcurve_data['seperated'] = {}
    lightcurve_data['medianmag'] = {}

    array_stars = numpy.asarray(lightcurve_data['stars'])

    lightcurve_data['count'] = {}
    lightcurve_data['median_mag_filters'] = {}

    # Build up the lists of indexed stars
    for i in range(0, len(median_ra)):
        check_in_index_array = numpy.where(clusters == i + 1)
        median_ra[i] = numpy.median(coord_list[0][check_in_index_array[0]])
        median_dec[i] = numpy.median(coord_list[1][check_in_index_array[0]])
        median_mag[i] = numpy.min(mag_list[check_in_index_array[0]])
        key = str(median_ra[i]) + " " + str(median_dec[i])
        lightcurve_data['seperated'][key] = []

        lightcurve_data['seperated'][key] = array_stars[check_in_index_array[0]]
        lightcurve_data['medianmag'][key] = median_mag[i]
        lightcurve_data['count'][key] = len(lightcurve_data['seperated'][key])
        lightcurve_data['median_mag_filters'][key] = {}
        for f in lightcurve_data['filters']:
            lightcurve_data['median_mag_filters'][key][f] = numpy.median(numpy.array(list(map(itemgetter('calibrated_magnitude'), filter(lambda x: x['filter'] == f, lightcurve_data['seperated'][key])))))

    user_choices = lightcurve_data['seperated'].keys()

    print('...Stars indexed!')

    return lightcurve_data, user_choices

#function to determine the exposure time equivalent for all of HC
def exp_equ(device=''):
	#sort out defaults
    if device=='': device=2
	
    class ServerTalkException(Exception):
        pass
    class NoStarsException(Exception):
        pass
	#setting user and passwd
    USER = 'sps_student'
    PASSWORD = 'sps_student'
	# Connect to the database
    try:
        print('Attempting to speak to server...')
        connection = pymysql.connect(host='129.12.24.29',
			user=USER,
			password=PASSWORD,
			db='imageportal',
			charset='utf8mb4',
			cursorclass=pymysql.cursors.DictCursor)
    except:
        raise ServerTalkException('...Failed to contact server!')
    print('...Connection established!')
    print('Retriving data from server...')
    with connection.cursor() as cursor:
		# Get all stars within radius
        sql = "SELECT o.id, o.fits_id, o.device_id, o.exptime, i.scale, i.mirror_diameter FROM observations as o, imaging_devices as i WHERE o.device_id = i.id;"
        cursor.execute(sql, ())
        image = cursor.fetchall()
    print('...data retrived!')
	
    total_exparea=0.0
    total_exparea_beacon=0.0
    beacon_count = 0
    for pic in image:
        scale_time=3600. #scales exp times to hours
        scale_size=1.0	#scales area to 1 square meter
        #only proceed if exposure time is set - images in metadata stage do not have that (set to None then)
        if (isinstance(pic['exptime'], float)):
            #print(pic)
            exp_area = float(pic['exptime'])/scale_time*(float(pic['mirror_diameter'])/scale_size)**2*numpy.pi/4.0
            total_exparea=total_exparea+exp_area
            if (pic['device_id']==device):
                total_exparea_beacon=total_exparea_beacon+exp_area
                beacon_count = beacon_count + 1
    print('Total Area * Exposure time: ',total_exparea,' hrs*m^2')
    print(total_exparea/ (math.pi*0.5*0.5),' Hours Exposure time on a 1m diameter telescope.')
    print('Device ID=',device,' Area * Exposure time: ',total_exparea_beacon,' hrs*m^2 (',total_exparea_beacon/total_exparea*100.0,'% )')
    print('Number of all Images: ',len(image))
    print('Number of ID=',device,' Images: ',beacon_count, ' (',100.0*beacon_count/len(image),'% )')

	#make a graph of uploaded image numbers or exposure time per time
    try:
        print('Attempting to speak to server...')
        connection = pymysql.connect(host='129.12.24.29',
                                     user=USER,
                                     password=PASSWORD,
                                     db='imageportal',
                                     charset='utf8mb4',
                                     cursorclass=pymysql.cursors.DictCursor)
    except:
        raise ServerTalkException('...Failed to contact server!')
    print('...Connection established!')
    print('Retriving data from server...')
    with connection.cursor() as cursor:
		# select all images and get their fits file upload time
		#sql = "SELECT o.id, o.fits_id, o.date, f.upload_time FROM observations as o, fits_files as f WHERE o.fits_id = f.id;"
                #get all images, their exposure time, their fits upload time and the device diameter
                sql = "SELECT o.id, o.fits_id, o.date, o.exptime, f.upload_time, i.mirror_diameter FROM observations as o, fits_files as f, imaging_devices as i WHERE (o.fits_id = f.id) and (o.device_id = i.id) and (o.device_id >= 0);"
                cursor.execute(sql, ())
                image = cursor.fetchall()
    print('...data retrived!')
    print(len(image))
    time_array=numpy.zeros(len(image))
    obstime_array=numpy.zeros(len(image))
    im_count=numpy.zeros(len(image))
    im_counter=numpy.zeros(len(image))
    exptime_array=numpy.zeros(len(image))
    mirror_array=numpy.zeros(len(image))
    area_array=numpy.zeros(len(image))
    counter=0
    for pic in image:
        time_array[counter] = pic['upload_time']
        obstime_array[counter] = pic['date']
        im_count[counter] = 1.0
        exptime_array[counter] = pic['exptime']
        mirror_array[counter] = pic['mirror_diameter']
        counter=counter+1

        ###################
        #add - sort this by the time array before continuing
    sorted = numpy.argsort(time_array)
    time_array = time_array[sorted]
    obstime_array = obstime_array[sorted]
    exptime_array = exptime_array[sorted]
    mirror_array = mirror_array[sorted]
    im_count = im_count[sorted]
    for i in range(0,len(image)):
        if (i == 0):
            im_counter[i] = im_count[i]
            area_array[i] = exptime_array[i] * mirror_array[i] * mirror_array[i] * numpy.pi / 4.0
        if (i >= 1):
            im_counter[i] = im_counter[i-1] + im_count[i]
            area_array[i] = area_array[i-1] + exptime_array[i] * mirror_array[i] * mirror_array[i] * numpy.pi / 4.0
	#normalise the time to days from seconds and use first day as start
    time_array = (time_array - numpy.min(time_array)) / (3600.0 * 24.0)
    print("Average image submission rate: " ,numpy.max(im_counter) / (numpy.max(time_array) / 365.25),"images per year")
    print("Average image submission rate: " ,numpy.max(im_counter) / (numpy.max(time_array) / 365.25 * 12.0),"images per month")
    print("Average image submission rate: " ,numpy.max(im_counter) / (numpy.max(time_array)),"images per day")

	#######################################
	#plot the submission date distributions
    rectangle1 = [0.15,0.1, 0.8,0.8]
    ax1 = plt.axes(rectangle1)
    plt.plot(time_array, im_counter, linestyle="solid", marker=".", markersize=0.01)
	#overplot average slope line
    ax1.plot([0,numpy.max(time_array)],[0,numpy.max(im_counter)], c="k", linestyle="--", alpha=0.75)
	#overplot time range for V1490cyg campaign
    rect = plt.Rectangle((342,0), 45, numpy.max(im_counter), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(364) , 0.95*numpy.max(im_counter) ,'V1490Cyg Campaign', horizontalalignment='center', verticalalignment='top', rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time range for V1491cyg campaign
    rect = plt.Rectangle((799,0), 30, numpy.max(im_counter), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(814) , 0.05*numpy.max(im_counter) ,'V1491Cyg Campaign', horizontalalignment='center', verticalalignment='bottom',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
    plt.xlabel("Days since Start of Database")
    plt.ylabel("Number of Images")
    ax1.set_ylim((0,numpy.max(im_counter)))
    ax1.set_xlim((0,numpy.max(time_array)))
	#plt.show()
    plt.savefig('data_submission_rate.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig('data_submission_rate.png', format='png', bbox_inches='tight', dpi=600)
    plt.clf()

        ###################
        #add - sort this by the observing time array before continuing
    sorted = numpy.argsort(obstime_array)
    time_array = time_array[sorted]
    obstime_array = obstime_array[sorted]
    exptime_array = exptime_array[sorted]
    mirror_array = mirror_array[sorted]
    im_count = im_count[sorted]
    for i in range(0,len(image)):
        if (i == 0):
            im_counter[i] = im_count[i]
            area_array[i] = exptime_array[i] * mirror_array[i] * mirror_array[i] * numpy.pi / 4.0 / 3600.0
        if (i >= 1):
            im_counter[i] = im_counter[i-1] + im_count[i]
            area_array[i] = area_array[i-1] + exptime_array[i] * mirror_array[i] * mirror_array[i] * numpy.pi / 4.0 / 3600.0
	
        #######################################
        #list of dates for overplot features in the plots
    alltalks = ['2020-09-15T18:00:00.0','2020-09-11T18:00:00.0','2020-09-04T18:00:00.0','2020-05-27T18:00:00.0','2020-05-15T18:00:00.0','2020-04-23T18:00:00.0','2020-02-15T18:00:00.0','2018-02-02T18:00:00.0','2019-05-02T18:00:00.0','2018-05-03T18:00:00.0','2018-07-03T18:00:00.0','2019-12-03T18:00:00.0','2019-07-04T18:00:00.0','2018-09-04T18:00:00.0','2019-04-05T18:00:00.0','2019-10-05T18:00:00.0','2018-07-07T18:00:00.0','2018-11-07T18:00:00.0','2018-12-07T18:00:00.0','2018-03-09T18:00:00.0','2020-01-10T18:00:00.0','2019-10-11T18:00:00.0','2019-01-12T18:00:00.0','2018-07-12T18:00:00.0','2018-07-13T18:00:00.0','2018-05-14T18:00:00.0','2019-05-14T18:00:00.0','2018-09-14T18:00:00.0','2019-06-15T18:00:00.0','2019-09-15T18:00:00.0','2019-04-17T18:00:00.0','2019-07-17T18:00:00.0','2019-06-19T18:00:00.0','2019-06-20T18:00:00.0','2019-04-23T18:00:00.0','2018-07-24T18:00:00.0','2018-08-24T18:00:00.0','2019-09-24T18:00:00.0','2018-11-24T18:00:00.0','2018-04-26T18:00:00.0','2018-10-26T18:00:00.0','2019-11-26T18:00:00.0','2018-04-27T18:00:00.0','2019-08-27T18:00:00.0','2018-03-28T18:00:00.0','2018-11-29T18:00:00.0','2018-11-30T18:00:00.0','2018-05-31T18:00:00.0','2019-05-31T18:00:00.0','2018-10-31T18:00:00.0','2018-10-31T18:00:00.0']
    alltalkstime = Time(alltalks, format='isot', scale='utc')
    allfirst = ['2020-01-01T00:00:00.0','2019-01-01T00:00:00.0','2018-01-01T00:00:00.0','2017-01-01T00:00:00.0','2016-01-01T00:00:00.0','2015-01-01T00:00:00.0','2014-01-01T00:00:00.0']
    allfirsttime = Time(allfirst, format='isot', scale='utc')
    allpapers = ['2020-08-01T00:00:00.0','2020-01-20T00:00:00.0','2018-12-15T00:00:00.0','2018-08-15T00:00:00.0','2018-06-15T00:00:00.0','2017-11-15T00:00:00.0','2017-04-15T00:00:00.0']
    allpaperstime = Time(allpapers, format='isot', scale='utc')
    allyearsnames = ['2019','2018','2017','2016','2015','2014']
    allyears = ['2019-07-01T00:00:00.1','2018-07-01T00:00:00.1','2017-07-01T00:00:00.1','2016-07-01T00:00:00.1','2015-07-01T00:00:00.1','2014-07-01T00:00:00.1']
    allyearstime = Time(allyears, format='isot', scale='utc')
	#######################################
	#plot the observation date distributions
        #only for data with sensible MJD 
    goodtimes = numpy.where(obstime_array >= 2000000.0)
    print("Average image observation rate: " ,numpy.max(im_counter[goodtimes[0]]) / ((numpy.max(obstime_array[goodtimes[0]])-numpy.min(obstime_array[goodtimes[0]])) / 365.25),"images per year")
    print("Average image observation rate: " ,numpy.max(im_counter[goodtimes[0]]) / ((numpy.max(obstime_array[goodtimes[0]])-numpy.min(obstime_array[goodtimes[0]])) / 365.25 * 12.0),"images per month")
    print("Average image observation rate: " ,numpy.max(im_counter[goodtimes[0]]) / ((numpy.max(obstime_array[goodtimes[0]])-numpy.min(obstime_array[goodtimes[0]]))),"images per day")
        #rescale imacounter to 1000 images
    im_counter = im_counter / 1000.0
    rectangle1 = [0.10,0.1, 0.85,0.85]
    ax1 = plt.axes(rectangle1)
    plt.plot(obstime_array[goodtimes[0]] - 2400000.5, im_counter[goodtimes[0]], linestyle="solid", marker=".", markersize=0.01)
	#overplot average slope line
    ax1.plot([numpy.min(obstime_array[goodtimes[0]] - 2400000.5),numpy.max(obstime_array[goodtimes[0]] - 2400000.5)],[0,numpy.max(im_counter[goodtimes[0]])], c="k", linestyle="--", alpha=0.75)
	#overplot time range for V1490cyg campaign
    rect = plt.Rectangle((58332.,0), 45, numpy.max(im_counter[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(58332.+45./2.) , 0.05*numpy.max(im_counter[goodtimes[0]]) ,'V1490Cyg Campaign', horizontalalignment='center', verticalalignment='bottom', rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time range for IC1396A campaign
    rect = plt.Rectangle((58655,0), 50, numpy.max(im_counter[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(58655+25) , 0.05*numpy.max(im_counter[goodtimes[0]]) ,'IC139A Campaign', horizontalalignment='center', verticalalignment='bottom',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time range for V1491cyg campaign
    rect = plt.Rectangle((58789,0), 30, numpy.max(im_counter[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(58789+15) , 0.05*numpy.max(im_counter[goodtimes[0]]) ,'V1491Cyg Campaign', horizontalalignment='center', verticalalignment='bottom',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time of start of database
    rect = plt.Rectangle((57988,0), 2, numpy.max(im_counter[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(57988+1) , 0.95*numpy.max(im_counter[goodtimes[0]]) ,'HOYS Database opens', horizontalalignment='center', verticalalignment='top',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time of start of beacon observations
    rect = plt.Rectangle((57266,0), 2, numpy.max(im_counter[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(57266+1) , 0.95*numpy.max(im_counter[goodtimes[0]]) ,'Beacon Observatory opens', horizontalalignment='center', verticalalignment='top',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time of start of HOYS project
    rect = plt.Rectangle((56956,0), 2, numpy.max(im_counter[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(56956+1) , 0.95*numpy.max(im_counter[goodtimes[0]]) ,'HOYS project starts', horizontalalignment='center', verticalalignment='top',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
        #overplot the HOYS talk dates
    for i in range(0,len(alltalkstime)):
        plt.plot([alltalkstime[i].mjd,alltalkstime[i].mjd],[0,0.03*numpy.max(im_counter[goodtimes[0]])],color='blue',linewidth=1)
        #overplot the HOYS paper publication dates
    for i in range(0,len(allpaperstime)):
        plt.plot([allpaperstime[i].mjd,allpaperstime[i].mjd],[0.97*numpy.max(im_counter[goodtimes[0]]),numpy.max(im_counter[goodtimes[0]])],color='green',linewidth=2,alpha=1)
	#overplot start of each calender year
    for i in range(0,len(allfirsttime)):
        plt.plot([allfirsttime[i].mjd,allfirsttime[i].mjd],[0,numpy.max(im_counter[goodtimes[0]])],color='blue',linewidth=1,linestyle='dashed',alpha=0.25)
	#overplot all the years on top of the plot
    for i in range(0,len(allyearstime)):
        plt.text( allyearstime[i].mjd, numpy.max(im_counter[goodtimes[0]]), allyearsnames[i], horizontalalignment='center', verticalalignment='bottom')
    plt.xlabel("MJD [days] of Observation")
    plt.ylabel("Number of Images [x 10$^3$]")
    ax1.set_ylim((0,numpy.max(im_counter[goodtimes[0]])))
    ax1.set_xlim((numpy.min(obstime_array[goodtimes[0]] - 2400000.5),numpy.max(obstime_array[goodtimes[0]] - 2400000.5)))
	#plt.show()
    plt.savefig('data_observation_rate.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig('data_observation_rate.png', format='png', bbox_inches='tight', dpi=600)
    plt.clf()

	#######################################
	#plot the observation date distributions for observing area times exptime
        #only for data with sensible MJD 
    goodtimes = numpy.where(obstime_array >= 2000000.0)
    print("Average image area rate: " ,numpy.max(area_array[goodtimes[0]]) / ((numpy.max(obstime_array[goodtimes[0]])-numpy.min(obstime_array[goodtimes[0]])) / 365.25),"hrs m2 per year")
    print("Average image area rate: " ,numpy.max(area_array[goodtimes[0]]) / ((numpy.max(obstime_array[goodtimes[0]])-numpy.min(obstime_array[goodtimes[0]])) / 365.25 * 12.0),"hrs m2 per month")
    print("Average image area rate: " ,numpy.max(area_array[goodtimes[0]]) / ((numpy.max(obstime_array[goodtimes[0]])-numpy.min(obstime_array[goodtimes[0]]))),"hrs m2 per day")
        #rescale the area array to 100 hr m^2
    area_array = area_array / 100.0
    rectangle1 = [0.10,0.1, 0.85,0.85]
    ax1 = plt.axes(rectangle1)
    plt.plot(obstime_array[goodtimes[0]] - 2400000.5, area_array[goodtimes[0]], linestyle="solid", marker=".", markersize=0.01)
	#overplot average slope line
    ax1.plot([numpy.min(obstime_array[goodtimes[0]] - 2400000.5),numpy.max(obstime_array[goodtimes[0]] - 2400000.5)],[0,numpy.max(area_array[goodtimes[0]])], c="k", linestyle="--", alpha=0.75)
	#overplot time range for V1490cyg campaign
    rect = plt.Rectangle((58332.,0), 45, numpy.max(area_array[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(58332.+45./2.) , 0.05*numpy.max(area_array[goodtimes[0]]) ,'V1490Cyg Campaign', horizontalalignment='center', verticalalignment='bottom', rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time range for IC1396A campaign
    rect = plt.Rectangle((58655,0), 50, numpy.max(area_array[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(58655+25) , 0.05*numpy.max(area_array[goodtimes[0]]) ,'IC139A Campaign', horizontalalignment='center', verticalalignment='bottom',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time range for V1491cyg campaign
    rect = plt.Rectangle((58789,0), 30, numpy.max(area_array[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(58789+15) , 0.05*numpy.max(area_array[goodtimes[0]]) ,'V1491Cyg Campaign', horizontalalignment='center', verticalalignment='bottom',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time of start of database
    rect = plt.Rectangle((57988,0), 2, numpy.max(area_array[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(57988+1) , 0.95*numpy.max(area_array[goodtimes[0]]) ,'HOYS Database opens', horizontalalignment='center', verticalalignment='top',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time of start of beacon observations
    rect = plt.Rectangle((57266,0), 2, numpy.max(area_array[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(57266+1) , 0.95*numpy.max(area_array[goodtimes[0]]) ,'Beacon Observatory opens', horizontalalignment='center', verticalalignment='top',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
	#overplot time of start of HOYS project
    rect = plt.Rectangle((56956,0), 2, numpy.max(area_array[goodtimes[0]]), angle=0.0, color="r", alpha=0.5)
    plt.text( numpy.max(56956+1) , 0.95*numpy.max(area_array[goodtimes[0]]) ,'HOYS project starts', horizontalalignment='center', verticalalignment='top',rotation=90.0,bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.75))
    ax1.add_patch(rect)
        #overplot the HOYS talk dates
    for i in range(0,len(alltalkstime)):
        	plt.plot([alltalkstime[i].mjd,alltalkstime[i].mjd],[0,0.03*numpy.max(area_array[goodtimes[0]])],color='blue',linewidth=1)
        #overplot the HOYS paper publication dates
    for i in range(0,len(allpaperstime)):
        	plt.plot([allpaperstime[i].mjd,allpaperstime[i].mjd],[0.97*numpy.max(area_array[goodtimes[0]]),numpy.max(area_array[goodtimes[0]])],color='green',linewidth=2,alpha=1)
	#overplot start of each calender year
    for i in range(0,len(allfirsttime)):
        	plt.plot([allfirsttime[i].mjd,allfirsttime[i].mjd],[0,numpy.max(area_array[goodtimes[0]])],color='blue',linewidth=1,linestyle='dashed',alpha=0.25)
	#overplot all the years on top of the plot
    for i in range(0,len(allyearstime)):
        	plt.text( allyearstime[i].mjd, numpy.max(area_array[goodtimes[0]]), allyearsnames[i], horizontalalignment='center', verticalalignment='bottom')
    plt.xlabel("MJD [days] of Observation")
    plt.ylabel("Total Exposure Time x Area [x 10$^2$ hr m$^2$]")
    ax1.set_ylim((0,numpy.max(area_array[goodtimes[0]])))
    ax1.set_xlim((numpy.min(obstime_array[goodtimes[0]] - 2400000.5),numpy.max(obstime_array[goodtimes[0]] - 2400000.5)))
	#plt.show()
    plt.savefig('data_area_rate.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig('data_area_rate.png', format='png', bbox_inches='tight', dpi=600)
    plt.clf()
        
	#exit()

	##############################################################################
	#also count the number of 'reliable' entries in the photometry table
    try:
        print('Attempting to speak to server...')
        connection = pymysql.connect(host='129.12.24.29',
			user=USER,
			password=PASSWORD,
			db='imageportal',
			charset='utf8mb4',
			cursorclass=pymysql.cursors.DictCursor)
    except:
        raise ServerTalkException('...Failed to contact server!')
    print('...Connection established!')
    print('Retriving data from server...')
    with connection.cursor() as cursor:
		# Get all stars within radius
        sql = "SELECT COUNT(*) FROM photometry WHERE flags<5 and calibrated_error > 0.0 and calibrated_error < 0.3 and calibrated_magnitude > 0.0 and calibrated_magnitude < 30.0"
        cursor.execute(sql, ())
        image = cursor.fetchall()
    print('...data retrived!')
    print('There are ',image[0]['COUNT(*)'],' reliable photometric datapoints')




    return(total_exparea,total_exparea_beacon,len(image))	

#function to simply plot a lightcurve
def make_plot_lightcurve(lightcurve_data,name='',err='',median='',dips='',filt_list='',symbols='',colours=''):
	#make some default symbol and colour definitions based on the filter
        #order of filters is Blue, Visual, Red, I-Band, H-alpha
    if err=='': err=0	#error bars on/off
    if median=='': median=0		#median filtered curve on/off
    if dips=='': dips=0	#highlight dips and burst points on/off
    if filt_list=='': filt_list=['B','V','R','I','HA']
    if symbols=='': symbols=['s','v','D','o','h']
    if colours=='': colours=['Blue','Green','Red','Black','Magenta']
    if name=='': name=lightcurve_data['name'][0]
    filt_list=numpy.array(filt_list)
    print('starting lightcurve plot...')
    rectangle1 = [0.1,0.1, 0.8,0.8]
    ax1 = plt.axes(rectangle1)
    minimum=30.0
    maximum=0.0
    i=0
    for fil in filt_list:
        check = numpy.where( (lightcurve_data['filter'] == fil) & (lightcurve_data['calibrated_magnitude'] >= 0) & (lightcurve_data['calibrated_magnitude'] <= 30) & (lightcurve_data['flags'] <= 4) & (lightcurve_data['calibrated_error'] > 0.0) & (lightcurve_data['calibrated_error'] < 0.3) )
        if (len(check[0] > 0)):
            plt.scatter(lightcurve_data[check[0]]['date'] - 2400000.5, lightcurve_data[check[0]]['calibrated_magnitude'], s=10.,c=colours[i], marker=symbols[i], edgecolor='black', alpha=1.0, lw=0.2)
            if (median == 1):
                plt.plot(lightcurve_data[check[0]]['date'] - 2400000.5, lightcurve_data[check[0]]['med_mag'], c=colours[i], marker=symbols[i], alpha=1.0, lw=0.2,linestyle='-', markersize=0)
            if (dips ==1):
                check_dip = numpy.where( (lightcurve_data['dips'] == -1) & (lightcurve_data['filter'] == fil) & (lightcurve_data['calibrated_magnitude'] >= 0) & (lightcurve_data['calibrated_magnitude'] <= 30) & (lightcurve_data['flags'] <= 4) & (lightcurve_data['calibrated_error'] > 0.0) & (lightcurve_data['calibrated_error'] < 0.3) )
                plt.scatter(lightcurve_data[check_dip[0]]['date'] - 2400000.5, lightcurve_data[check_dip[0]]['calibrated_magnitude'], s=20.,c=colours[i], marker=symbols[i], edgecolor='red', alpha=1.0, lw=1)
                check_burst = numpy.where( (lightcurve_data['dips'] == 1) & (lightcurve_data['filter'] == fil) & (lightcurve_data['calibrated_magnitude'] >= 0) & (lightcurve_data['calibrated_magnitude'] <= 30) & (lightcurve_data['flags'] <= 4) & (lightcurve_data['calibrated_error'] > 0.0) & (lightcurve_data['calibrated_error'] < 0.3) )
                plt.scatter(lightcurve_data[check_burst[0]]['date'] - 2400000.5, lightcurve_data[check_burst[0]]['calibrated_magnitude'], s=20.,c=colours[i], marker=symbols[i], edgecolor='green', alpha=1.0, lw=1)
            if (err == 1):
				#plt.errorbar(lightcurve_data[check[0]]['date'], lightcurve_data[check[0]]['calibrated_magnitude'], yerr=lightcurve_data[check[0]]['calibrated_error'], c=colours[i], marker='', alpha=1.0, lw=0.2, linestyle='')
                plt.errorbar(lightcurve_data[check[0]]['date'] - 2400000.5, lightcurve_data[check[0]]['calibrated_magnitude'], yerr=lightcurve_data[check[0]]['org_cal_err'], c=colours[i], marker='', alpha=1.0, lw=0.2, linestyle='')
            maximum2 = numpy.max(lightcurve_data[check[0]]['calibrated_magnitude'])
            minimum2 = numpy.min(lightcurve_data[check[0]]['calibrated_magnitude'])
            if (maximum2 > maximum): maximum=maximum2
            if (minimum2 < minimum): minimum=minimum2
        i=i+1
    ax1.set_ylabel('Magnitudes')
    ax1.set_xlabel('Modified Julian Date')
    ax1.set_ylim((maximum+0.3,minimum-0.5))
	#ax1.set_xlim((2458335.0,2458380.0))
    plt.text( numpy.min(lightcurve_data['date'] - 2400000.5) , minimum-0.2 ,lightcurve_data[0]['name'], horizontalalignment='left',bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        #plt.text( 2458379.0 , minimum-0.2,lightcurve_data[0]['name'], horizontalalignment='right',bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
	#plt.show()
    plt.savefig('lightcurves/lightcurve_'+name+'.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig('lightcurves/lightcurve_'+name+'.png', format='png', bbox_inches='tight', dpi=600)
    plt.clf()
    return()

#function to simply plot a colour curve
def make_plot_colourcurve(lightcurve,name='',m1='',m2='',err='',color='',size='',marker=''):
	#make some default symbol and colour definitions based on the filter
        #order of filters is Blue, Visual, Red, I-Band, H-alpha
    if err=='': err=0	#error bars on/off
    if m1=='': m1='V'	#default the first mag in the colour
    if m2=='': m2='I'	#default the first mag in the colour
    if color=='': color='green'
    if size=='': size=5
    if marker=='': marker='o'
    if name=='': name=lightcurve_data['name'][0]
    print('starting colourcurve plot...')
    rectangle1 = [0.1,0.1, 0.8,0.8]
    ax1 = plt.axes(rectangle1)
    minimum=30.0
    maximum=-30.0
    check = numpy.where( (lightcurve[m1] <= 30 ) & (lightcurve[m2] <= 30) & (lightcurve[m1] > -10) & (lightcurve[m2] > -10) & (lightcurve['calibrated_magnitude'] >= 0) & (lightcurve['calibrated_magnitude'] <= 30) & (lightcurve['flags'] <= 4) & (lightcurve['calibrated_error'] > 0.0) & (lightcurve['calibrated_error'] < 0.3) )
    if (len(check[0] > 0)):
        plt.scatter(lightcurve[check[0]]['date'], lightcurve[check[0]][m1]-lightcurve[check[0]][m2], s=size, c=color, marker=marker, edgecolor='black', alpha=1.0, lw=0.2)
        if (err == 1):
            plt.errorbar(lightcurve[check[0]]['date'], lightcurve[check[0]][m1]-lightcurve[check[0]][m2], yerr=numpy.sqrt(numpy.square(lightcurve[check[0]][m1+'e']) + numpy.square(lightcurve[check[0]][m2+'e'])), c=color, marker='', alpha=1.0, lw=0.2, linestyle='')
        maximum2 = numpy.max(lightcurve[check[0]][m1]-lightcurve[check[0]][m2])
        minimum2 = numpy.min(lightcurve[check[0]][m1]-lightcurve[check[0]][m2])
        if (maximum2 > maximum): maximum=maximum2
        if (minimum2 < minimum): minimum=minimum2
    ax1.set_ylabel('Colour ('+m1+'-'+m2+') [mag]')
    ax1.set_xlabel('Julian Date')
    ax1.set_ylim((minimum-0.05,maximum+0.05))
	#plt.show()
    plt.savefig('lightcurves/colorcurve_'+name+'.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig('lightcurves/colorcurve_'+name+'.png', format='png', bbox_inches='tight', dpi=600)
    plt.clf()
    return()

#function to simply plot a colour magnitude curve
def make_plot_cmd(lightcurve,name,m1='',c1='',c2='',err='',color='',size='',marker=''):
	#make some default symbol and colour definitions based on the filter
        #order of filters is Blue, Visual, Red, I-Band, H-alpha
    if err=='': err=0	#error bars on/off
    if m1=='': m1='V'	#default the mag
    if c1=='': c1='V'	#default the first mag in the colour
    if c2=='': c2='I'	#default the first mag in the colour
    if color=='': color='green'
    if size=='': size=5
    if marker=='': marker='s'
    print('starting cmd plot...')
    rectangle1 = [0.1,0.1, 0.8,0.8]
    ax1 = plt.axes(rectangle1)
    minimumx=30.0
    maximumx=-30.0
    minimumy=30.0
    maximumy=0.0
    check = numpy.where( (lightcurve[m1] <= 30 ) & (lightcurve[c1] <= 30) & (lightcurve[c2] <= 30) & (lightcurve[m1] > -10) & (lightcurve[c1] > -10)  & (lightcurve[c2] > -10)& (lightcurve['calibrated_magnitude'] >= 0) & (lightcurve['calibrated_magnitude'] <= 30) & (lightcurve['flags'] <= 4) & (lightcurve['calibrated_error'] > 0.0) & (lightcurve['calibrated_error'] < 0.3) )
    if (len(check[0] > 0)):
        plt.scatter(lightcurve[check[0]][c1]-lightcurve[check[0]][c2], lightcurve[check[0]][m1], s=size, c=color, marker=marker, edgecolor='black', alpha=1.0, lw=0.2)
        if (err == 1):
            plt.errorbar(lightcurve[check[0]][c1]-lightcurve[check[0]][c2], lightcurve[check[0]][m1], yerr=lightcurve[check[0]][m1+'e'], xerr=numpy.sqrt(numpy.square(lightcurve[check[0]][c1+'e']) + numpy.square(lightcurve[check[0]][c2+'e'])), c=color, marker='', alpha=1.0, lw=0.2, linestyle='')
            maximumx2 = numpy.max(lightcurve[check[0]][c1]-lightcurve[check[0]][c2])
            minimumx2 = numpy.min(lightcurve[check[0]][c1]-lightcurve[check[0]][c2])
            if (maximumx2 > maximumx): maximumx=maximumx2
            if (minimumx2 < minimumx): minimumx=minimumx2
            maximumy2 = numpy.max(lightcurve[check[0]][m1])
            minimumy2 = numpy.min(lightcurve[check[0]][m1])
            if (maximumy2 > maximumy): maximumy=maximumy2
            if (minimumy2 < minimumy): minimumy=minimumy2
    ax1.set_xlabel('Colour ('+c1+'-'+c2+') [mag]')
    ax1.set_ylabel('Magnitude ('+m1+') [mag]')
    ax1.set_ylim((maximumy+0.30,minimumy-0.30))
    ax1.set_xlim((minimumx-0.05,maximumx+0.05))
	#plt.show()
    plt.savefig('lightcurves/cmd_'+name+'.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig('lightcurves/cmd_'+name+'.png', format='png', bbox_inches='tight', dpi=600)
    plt.clf()


def dip_detection_probability(sourcename,plot=''):
	#sort out the defaults
    if (plot==''): plot = 0		#do not make the plot as default
	#read in a particular lightcurve
    lightcurve1=numpy.load('lc_data/'+sourcename)
    color=['green','red','black']
    filt = ['V','R','I']
    marker=['v','D','o']
	
    if (plot==1):
        rectangle1 = [0.1,0.1, 0.8,0.8]
        ax1 = plt.axes(rectangle1)
	#make an array of durations in days
    durations = numpy.arange(0,100,1) #[1,2,4,8,16,32]
	#then make a probability array of same length
    prob_array=numpy.zeros((len(filt),len(durations)))
	
	#loop over al three filters
    for f in range(0,len(filt)):
	
		#get the I-band data - to test the procedure
        lightcurve=lightcurve1[numpy.where(lightcurve1['filter']==filt[f])]
        if (len(lightcurve) > 10):
	        
			#make a time array that is as long as the lightcurve but with much smaller, and equal sized steps
            time_array = numpy.arange(numpy.min(lightcurve['date']),numpy.max(lightcurve['date']),)
            obs_dates=lightcurve['date']
		
			#now loop over the durations
            for i in range(0,len(durations)):
                dur=durations[i]
			        #make a cover array
                cover_array = numpy.zeros(len(time_array))
			        #find all the times in time_array where the date points are more than 0.5 away from an actual datapoint and ID them in cover array
                for j in range(0,len(obs_dates)):
                    check=numpy.where( numpy.abs(time_array - obs_dates[j]) < dur/2. )
			                #print(check[0])
                    cover_array[check[0]]=1
				#find al the remaining zeros in cover_array (thats the bits where the dip can 'hide')
                check=numpy.where(cover_array==0)
			        #fill in probability array
                prob_array[f,i]=100.0*(1.0-float(len(check[0]))/float(len(time_array)))
	
        if (plot==1):
            plt.plot(durations,prob_array[f,:],color=color[f],marker=marker[f], alpha=1.0, lw=1,linestyle='solid', markersize=0)
	
    if (plot==1):
        ax1.set_ylabel('Detection Probability [%]')
        ax1.set_xlabel('Duration of Dips [days]')
        ax1.set_xlim((0,numpy.max(durations)))
        ax1.set_ylim((0,100))
        plt.text( numpy.max(durations)*0.98 , 4 ,'Target: '+sourcename+'\nRegion: '+str(lightcurve1['target'][0]), horizontalalignment='right',bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
		#plt.show()
        plt.savefig('dip_detection_prob.pdf', format='pdf', bbox_inches='tight', dpi=600)
        plt.savefig('dip_detection_prob.png', format='png', bbox_inches='tight', dpi=600)
        plt.clf()
    return(prob_array)


#function that determines for each datapoint the magnitudes in the other filters
#as the median in a given timeframe - essentially a bit like the median filter
#function, but for all filters to allow the colours to be determined and the 
#default filter size is only one day.
def medmag_time(lightcurve,filt_list='',filt_time=''):
	#sort out the defaults
    if filt_time=='': filt_time=[1,1,1,1,1,1]
    if filt_list=='': filt_list=['U','B','V','R','I','HA']
	#loop over all datapoints
    for i in range(0,len(lightcurve)):
		#loop over all the filters
        for f in range(0,len(filt_list)):
                	#find all the datapoints in the filter on the same day/in relevant time range
                    check = numpy.where( (lightcurve['filter'] == filt_list[f]) &  (numpy.abs(lightcurve[i]['date'] - lightcurve['date']) < filt_time[f]) & (lightcurve['calibrated_magnitude'] >= 0) & (lightcurve['calibrated_magnitude'] <= 30) & (lightcurve['flags'] <= 4) & (lightcurve['calibrated_error'] > 0.0) & (lightcurve['calibrated_error'] < 0.3) )
                    if len(check[0] > 0):
                        	#and determine the median value
                            lightcurve[i][filt_list[f]] = numpy.median(lightcurve[check[0]]['calibrated_magnitude'])
                                #and determine the uncertainties
                            lightcurve[i][filt_list[f]+'e'] = numpy.sqrt(numpy.sum(numpy.square(lightcurve[check[0]]['calibrated_error']))/len(check[0]))
    return(lightcurve)
                        	

#function that median filters lightcurves over a given time interval
def medfilter_time_org(lightcurve,filt_list='',filt_time=''):
	#sort out the defaults
    if filt_list=='': filt_list=['U','B','V','R','I','HA']
    if filt_time=='': filt_time=[150,150,150,150,150,300]
        #loop over all the filters
    counter = 0
    for j in range(0,len(filt_list)):
		#extract only the single filter
        single_lightcurve = extract_filter(lightcurve,filt_list[j])
        if (len(single_lightcurve) > 0):
        		#loop over all the datapoints
                for i in range(0,len(single_lightcurve)):
                    check = numpy.where( (numpy.abs(single_lightcurve[i]['date'] - single_lightcurve['date'])  < filt_time[j])   & (single_lightcurve['calibrated_magnitude'] >= 0) & (single_lightcurve['calibrated_magnitude'] <= 30) & (single_lightcurve['flags'] <= 4) & (single_lightcurve['calibrated_error'] > 0.0) & (single_lightcurve['calibrated_error'] < 0.3) )
                    if len(check[0] > 0):
                        single_lightcurve[i]['med_mag'] = numpy.median(single_lightcurve[check[0]]['calibrated_magnitude'])
			#merge the individual filtered lightcurves back together
                if (counter == 0):
                    end_lightcurve = single_lightcurve
                if (counter > 0):
                    end_lightcurve = numpy.concatenate((end_lightcurve,single_lightcurve))
                counter = counter+1
    if (counter > 0):
        end_lightcurve = sort_time(end_lightcurve)
        return(end_lightcurve)
    if (counter == 0):
        return(lightcurve)

def medfilter_time(lightcurve,filt_list='',filt_time=''):
	#sort out the defaults
    if filt_list=='': filt_list=['U','B','V','R','I','HA']
    if filt_time=='': filt_time=[150,150,150,150,150,300]
    #loop over all the filters
    counter = 0
    for j in range(0,len(filt_list)):
		#extract only the single filter
        single_lightcurve = extract_filter(lightcurve,filt_list[j])
        #make a homogeneously sampled 'median' array for the time
        med_time_array = numpy.arange(numpy.min(single_lightcurve['date']),numpy.max(single_lightcurve['date']),1)
        #and an corresponding one for the magnitudes
        med_mag_array = numpy.array(med_time_array*0.0)
        med_mag_array_filtered = numpy.array(med_time_array*0.0)
        #fill this array with a median filter over XXX days
        XXX=2.0*filt_time[j] # 10.0
        for i in range(0,len(med_time_array)):
            check = numpy.where( (numpy.abs(med_time_array[i] - single_lightcurve['date']) < XXX/2.0)  & (single_lightcurve['dips'] == 0) & (single_lightcurve['calibrated_magnitude'] >= 0) & (single_lightcurve['calibrated_magnitude'] <= 30) & (single_lightcurve['flags'] <= 4) & (single_lightcurve['calibrated_error'] > 0.0) & (single_lightcurve['calibrated_error'] < 0.3))
            med_mag_array[i] = numpy.nanmedian( single_lightcurve[check[0]]['calibrated_magnitude'] )
                #now filter this with the given time span
            for i in range(0,len(med_time_array)):
                check = numpy.where( numpy.abs(med_time_array[i] -  med_time_array) < filt_time[j])
                med_mag_array_filtered[i] = numpy.nanmedian(med_mag_array[check[0]])
        if (len(single_lightcurve) > 0):
        		#loop over all the datapoints and fill in the nearest filtered magnitude
                for i in range(0,len(single_lightcurve)):
				#determine time differences to all med_time_array dates
                                diff = numpy.abs(single_lightcurve[i]['date']-med_time_array)
                                sort = numpy.argsort(diff)
                                single_lightcurve[i]['med_mag'] = med_mag_array_filtered[sort[0]]
			#merge the individual filtered lightcurves back together
                if (counter == 0):
                    end_lightcurve = single_lightcurve
                if (counter > 0):
                    end_lightcurve = numpy.concatenate((end_lightcurve,single_lightcurve))
                counter = counter+1
    if (counter > 0):
        end_lightcurve = sort_time(end_lightcurve)
        return(end_lightcurve)
    if (counter == 0):
        return(lightcurve)

#function that identifies dips and outbursts for a lightcurve
#the 'dip' parameter in the datastructure is set to -1 if the curve is in a dip, +1 if it is in an outburst and 0 otherwise
def find_dips_org(lightcurve,filt_list='',filt_time='',sigma='',maxit=''):
	#start by sorting out the defaults
    if filt_list=='': filt_list=['U','B','V','R','I','HA']
    if filt_time=='': filt_time=[150,150,150,150,150,300]
    if sigma=='': sigma=3.0
    if maxit=='': maxit=10
        #reset all the 'dips' values to zero in case they have been processed before
    lightcurve['dips'] = 0.0
	#extract and median filter only the single filter
    median_lightcurve = medfilter_time(lightcurve,filt_list,filt_time)
        #loop over all the filters
    counter=0
    for i in range(0,len(filt_list)):
		#extract and median filter only the single filter
        single_lightcurve = extract_filter(median_lightcurve,filt_list[i])
                #only proceed if there are more than 10 datapoints in the lightcurve
        if (len(single_lightcurve) > 10):
                        #make a copy of the single filter lightcurve
                        medrem_lc = numpy.array(single_lightcurve)
                        #remove the median from the lightcurve
                        medrem_lc['calibrated_magnitude']=medrem_lc['calibrated_magnitude']-medrem_lc['med_mag']
			#now iteratively remove 'sigma' sigma outliers above and below the median subtracted lightcurve
                        #until either no further points are removed or NN iterations have been done
                        iteration=0
                        new_dips=1000
                        while( (iteration<=maxit) and (new_dips!=0)):
				#identify the datapoints which are not in dips or outbursts
                                check = numpy.where( (medrem_lc['dips']==0))
                                #determine how many points are in dips+bursts
                                num_dips = len(medrem_lc)-len(check[0])
                                #determine the rms of the median removed lightcurve but only for non dip/burst points
                                rms = numpy.std(medrem_lc[check[0]]['calibrated_magnitude'])
                                #identify dip datapoints
                                check_dip = numpy.where( medrem_lc['calibrated_magnitude'] / rms >= sigma )
                                #identify outburst datapoints
                                check_burst = numpy.where( medrem_lc['calibrated_magnitude'] / rms <= -sigma )
                                #rewrite the 'dips' variable for all the dips and burst points
                                medrem_lc['dips'][check_dip[0]]= -1.0
                                medrem_lc['dips'][check_burst[0]] = +1.0
                                #determine how many new dips+burst have been identified
                                new_dips=len(check_dip[0])+len(check_burst[0])-num_dips
				#count up the iteration count
                                iteration=iteration+1
                        print('Found dips in ',filt_list[i],' after ',iteration,' iterations')
                        #write the dip variable back into the original non-altered lightcurve
                        single_lightcurve['dips']=medrem_lc['dips']
                        #put together the individual lightcurves into one datastructure
                        if (counter == 0):
                            end_lightcurve = single_lightcurve
                        if (counter > 0):
                            end_lightcurve = numpy.concatenate((end_lightcurve,single_lightcurve))
                        counter = counter+1
                        
	#if there are no dips/bursts fount then nothing has changed, hence
    if (counter == 0):
        end_lightcurve = lightcurve
        end_lightcurve = sort_time(end_lightcurve)
        
        #make_plot_lightcurve(end_lightcurve,'test',err=1,median=1,dips=1)
	#return lightcurve with dip parameter changes as well as array of number of dips and burst with their properties
    return(end_lightcurve)

def find_dips(lightcurve,filt_list='',filt_time='',sigma='',maxit=''):
	#start by sorting out the defaults
    if filt_list=='': filt_list=['U','B','V','R','I','HA']
    if filt_time=='': filt_time=[150,150,150,150,150,300]
    if sigma=='': sigma=3.0
    if maxit=='': maxit=10
    #reset all the 'dips' values to zero in case they have been processed before
    lightcurve['dips'] = 0
        #loop over all the filters
    counter=0
    for i in range(0,len(filt_list)):
		#extract and median filter only the single filter
        single_lightcurve = extract_filter(lightcurve,[filt_list[i]])
                #only proceed if there are more than 10 datapoints in the lightcurve
        if (len(single_lightcurve) > 10):
			#now iteratively remove 'sigma' sigma outliers above and below the median subtracted lightcurve
                        #until either no further points are removed or NN iterations have been done
                        iteration=0
                        new_dips=1000
                        while( (iteration<=maxit) and (new_dips!=0)):
                                #median filter this one
                                single_lightcurve = medfilter_time(single_lightcurve,filt_list[i],[filt_time[i]])
                                #identify the datapoints which are not in dips or outbursts
                                check = numpy.where( (single_lightcurve['dips']==0))
                                #determine how many points are in dips+bursts
                                num_dips = len(single_lightcurve)-len(check[0])
                                #determine the rms of the median removed lightcurve but only for non dip/burst points
                                rms = numpy.nanstd(single_lightcurve[check[0]]['calibrated_magnitude']-single_lightcurve[check[0]]['med_mag'])
                                #identify dip datapoints
                                check_dip = numpy.where( (single_lightcurve['calibrated_magnitude']-single_lightcurve['med_mag']) / rms >= sigma )
                                #identify outburst datapoints
                                check_burst = numpy.where( (single_lightcurve['calibrated_magnitude']-single_lightcurve['med_mag']) / rms <= -sigma )
                                #rewrite the 'dips' variable for all the dips and burst points
                                single_lightcurve['dips'][check_dip[0]]= -1.0
                                single_lightcurve['dips'][check_burst[0]] = +1.0
                                #determine how many new dips+burst have been identified
                                new_dips=len(check_dip[0])+len(check_burst[0])-num_dips
				#count up the iteration count
                                iteration=iteration+1
                        print('Found ',len(check_dip[0]),' points in dips and ',len(check_burst[0]),' points in bursts in ',filt_list[i],' after ',iteration,' iterations')
                        #put together the individual lightcurves into one datastructure
                        if (counter == 0):
                            end_lightcurve = single_lightcurve
                        if (counter > 0):
                            end_lightcurve = numpy.concatenate((end_lightcurve,single_lightcurve))
                        counter = counter+1
                        
	#if there are no dips/bursts fount then nothing has changed, hence
    if (counter == 0):
        end_lightcurve = lightcurve
        end_lightcurve = sort_time(end_lightcurve)
        
        #make_plot_lightcurve(end_lightcurve,'test',err=1,median=1,dips=1)
	#return lightcurve with dip parameter changes as well as array of number of dips and burst with their properties
    return(end_lightcurve)

#funtion that goes through a lightcurve and determines the properties of all the dips/bursts determined earlier
#it will first establish the number of dips/bursts, measure their length, depth etc.
#it will return an array of all the dips/bursts and all their properties as a dip data structure
def determine_dip_properties(lightcurve,filt_list=''):
	#start by sorting out the defaults
    if filt_list=='': filt_list=['U','B','V','R','I','HA']
        #reset all the 'dips_num' values to zero in case they have been processed before
    lightcurve['dips_num'] = 0.0
	#############################################
	#at first just count all the dips and bursts
        #############################################
        #loop over all the filters
    counter=0
    dip_counter=1 #has to start from 1
    for i in range(0,len(filt_list)):
		#extract only the single filter
        single_lightcurve = extract_filter(lightcurve,filt_list[i])
               	#sort lightcurve by time, in case it's not
        single_lightcurve = sort_time(single_lightcurve)
                #process the dips and bursts
        check_dips = numpy.where(single_lightcurve['dips'] == -1)
        check_bursts = numpy.where(single_lightcurve['dips'] == +1)
		#process the dips
        if (len(check_dips[0]) > 0):
                	#print('There are ',len(check_dips[0]),'dips in',filt_list[i])
                        #loop over all datapoints that are in a dip
                        for d in check_dips[0]:
                        	#set the 'dip_num' value to the current counter
                            single_lightcurve['dips_num'][d] = dip_counter
                                #check if d is at the end of the lightcurve, and if not and the next point is not in a dip, increase the dip counter
                        	#print(d, numpy.max(check_dips[0]))
                            if ( d < len(single_lightcurve) and ( d+1 not in check_dips[0])):
                                dip_counter = dip_counter+1
                            if ( d == len(single_lightcurve)):
                                dip_counter = dip_counter+1
                    
                #process the bursts
        if (len(check_bursts[0]) > 0):
                	#print('There are ',len(check_bursts[0]),'bursts in',filt_list[i])
                        #loop over all datapoints that are in a burst
                        for d in check_bursts[0]:
                        	#set the 'dip_num' value to the current counter
                            single_lightcurve['dips_num'][d] = dip_counter
                                #check if d is at the end of the lightcurve, and if not and the next point is not in a dip, increase the dip counter
                            if ( d < len(single_lightcurve) and ( d+1 not in check_bursts[0])):
                                    dip_counter = dip_counter+1
                            if ( d == len(single_lightcurve)):
                                    dip_counter = dip_counter+1

		#put together the individual lightcurves into one datastructure
        if (counter == 0):
            end_lightcurve = single_lightcurve
        if (counter > 0):
            end_lightcurve = numpy.concatenate((end_lightcurve,single_lightcurve))
        counter = counter+1

	#decrease the dip counter since the very last increase is not needed
        if (dip_counter != 1):
            dip_counter = dip_counter-1

        print('There are ',dip_counter,' potential dips and bursts in the lightcurve.')

       ###########################################
        #now generate a dip/burst array and determine all the properties
        ##########################################

	#define the dip/burst datastructure
        diptype = numpy.dtype([('name',str,80),		#name of source
         		                ('number',int),		#number of the dip
        				('filter',str,80),		#filter the dip is occuring in
                                        ('dip_burst',int),		#-1 if its a dip, +1 if its a burst
                                        ('num_points',int),		#number of datapoint in the dip/burst
                                        ('min_duration',float),		#minimum duration in days i.e. difference between 1st and last datapoint in dip/burst
                                       ('jd_mid_min_duration',float),	#middle point in JD of the above
                                        ('max_duration',float),		#maximum duration in days i.e. difference between 1st and last datapoint outside dip/burst
                                       ('jd_mid_max_duration',float),	#middle point in JD of the above
                                       ('ave_depth',float),		#average depth if dip/hight of burst in mag in filter used, positive for dips, negative for bursts
                                       ('med_depth',float),		#median depth if dip/hight of burst in mag in filter used, positive for dips, negative for bursts
                                       ('target',int),				#ID of the target field
                                       ('free',float),				#free variable for uses in prcessing later on
        				])

	#make dip array and set all values to zero
        dip_array = numpy.empty(dip_counter,dtype=diptype)
        
        if (dip_counter > 1):
	        
	        #now loop over all the dips and fill in the dip datastructure and fill in the dip properties
            for i in range(0,dip_counter):
	                #print('determining properties of dip/burst number ',i)
                        #fill in the free parameter
                        dip_array['free'][i]=0
	        	#fill in the dip number
                        dip_array['number'][i]=i+1
	                #get all the datapoints for this particular dip/burst
                        check_dip = numpy.where(end_lightcurve['dips_num'] == i+1)
	        	#fill in the name of the source
                        dip_array['name'][i]=end_lightcurve[check_dip[0][0]]['name']
	        	#fill in the name of the target
                        dip_array['target'][i]=end_lightcurve[check_dip[0][0]]['target']
	                #fill in the dip filter
                        dip_array['filter'][i] = end_lightcurve[check_dip[0][0]]['filter']
	                #fill in if it is dip or burst
                        dip_array['dip_burst'][i] = end_lightcurve[check_dip[0][0]]['dips']
	                #fill in number of datapoints in dip/burst
                        dip_array['num_points'][i] = len(check_dip[0])
	                #fill in the average and median depth of the dip
                        dip_array['ave_depth'][i] = numpy.mean(end_lightcurve[check_dip[0]]['calibrated_magnitude']-end_lightcurve[check_dip[0]]['med_mag'])
                        dip_array['med_depth'][i] = numpy.median(end_lightcurve[check_dip[0]]['calibrated_magnitude']-end_lightcurve[check_dip[0]]['med_mag'])
	
	                #fill in min/max duration
                        help_lc=extract_filter(end_lightcurve,dip_array['filter'][i])
                        help_lc=sort_time(help_lc)
                        check_help=numpy.where(help_lc['dips_num'] == i+1)
                        min_i=numpy.min(check_help[0])
                        min_jd = help_lc['date'][min_i]
                        max_i=numpy.max(check_help[0])
                        max_jd = help_lc['date'][max_i]
                        dip_array['min_duration'][i] = max_jd-min_jd
                        dip_array['jd_mid_min_duration'][i] = (max_jd+min_jd)/2.0
                        if (min_i >0): 
                            min_jd = help_lc['date'][min_i -1]
                        if (max_i+1 < len(help_lc)):
                            max_jd = help_lc['date'][max_i+1]
                        dip_array['max_duration'][i] = max_jd-min_jd
                        dip_array['jd_mid_max_duration'][i] = (max_jd+min_jd)/2.0
	
		#print out the dip properties to the screen 
                #print('Original Dip Array')
                #print(dip_array)
            dip_array=extract_real_dips(dip_array,len_dip=1)
            print('Dip Array after removal of spurious dips')
            print(dip_array)
            if (dip_array == 0):
                return(end_lightcurve,0)
		
                #if dips have potentially been removed, the remove those entries in the lightcurve
                #make list of all original dip numbers
                list_num_org=numpy.unique(end_lightcurve['dips_num'])
                list_num_remain=numpy.unique(dip_array['number'])
                #find which ones are not there anymore and remove the entries in the 'dips' parameters
                for i in list_num_org:
                    if not i in list_num_remain:
                        check_overwrite=numpy.where(end_lightcurve['dips_num'] == i)
                        end_lightcurve['dips_num'][check_overwrite[0]]=0
                        end_lightcurve['dips'][check_overwrite[0]]=0

        print('There are ',len(dip_array[numpy.where((dip_array['dip_burst'] == -1) & (dip_array['filter']=='V'))]),' real Dips in V.')
        print('There are ',len(dip_array[numpy.where((dip_array['dip_burst'] == -1) & (dip_array['filter']=='R'))]),' real Dips in R.')
        print('There are ',len(dip_array[numpy.where((dip_array['dip_burst'] == -1) & (dip_array['filter']=='I'))]),' real Dips in I.')
        print('There are ',len(dip_array[numpy.where((dip_array['dip_burst'] == +1) & (dip_array['filter']=='V'))]),' real Bursts in V.')
        print('There are ',len(dip_array[numpy.where((dip_array['dip_burst'] == +1) & (dip_array['filter']=='R'))]),' real Bursts in R.')
        print('There are ',len(dip_array[numpy.where((dip_array['dip_burst'] == +1) & (dip_array['filter']=='I'))]),' real Bursts in I.')
        return(end_lightcurve, dip_array)
    return(lightcurve,0)

#have a function that returns only dips cosidered to be 'real' i.e. longer than a given time or having a counterpart dip/burst at different filter
#within a day +- duration of the dip
def extract_real_dips(dips,len_dip=''):
	#sort out the defaults
    if (len_dip==''): lendip=1
        #check which dip is real in the list and remove all the other ones
        #make arrary of length dips
    real = numpy.zeros(len(dips))
        #extract all the dips with more than len_dip and ID as real
    check=numpy.where(dips['num_points'] > len_dip)
    real[check[0]]=1
        #go through all the other dips and make them real if there is another dip in a different filter within one day+-min_length
    check=numpy.where(dips['num_points'] <= len_dip)
        #loop over all those dips
    for i in range(0,len(check[0])):
        	#get the dip number for ID reasons
            num=dips[check[0][i]]['number']
                #extract all other dips in the other filters which are also a dip/burst and within one day
            check2=numpy.where( (dips['filter'] != dips[num-1]['filter']) & (dips['dip_burst'] == dips[num-1]['dip_burst']) & ( numpy.abs(dips[num-1]['jd_mid_min_duration']-dips['jd_mid_min_duration']) < 1.0+dips[num-1]['min_duration'] ) )
                #only continue if there are at least one other (i.e. more than one selected)
            if (len(check2[0])>0):
                        #label the dip as real
                    real[num-1]=1
	#now remove all the entries in dips which are not real
    check=numpy.where(real==1)
    if (len(check[0]) > 0):
        dips=dips[check[0]]
        return(dips)
    return(0)

#function to sort lightcurve data by time
def sort_time(lightcurve):
        return(lightcurve[numpy.argsort(lightcurve['date'])])

#function to ....
def extract_userid(lightcurve,value,id_list='',invert=''):
	#start by sorting out the defaults
    if id_list=='': id_list=[7] #defaulted to only extract user 7
    if invert=='': invert=0 #0 being true, 1 being false, i.e. extract turns into remove
    return(lightcurve[ numpy.isin(lightcurve[value],id_list,invert=invert) ])

#function to extract data per given parameter (e.g. user_id, filter, etc.)
def extract_data(lightcurve,prop_name='',id_list='',invert=''):
	#start by sorting out the defaults
    if prop_name=='': prop_name='user_id' #defaulted to work on user_id
    if id_list=='': id_list=[7] #defaulted to only extract user 7
    if invert=='': invert=0 #0 being true, 1 being false, i.e. extract turns into remove
    return(lightcurve[numpy.isin(lightcurve[prop_name],id_list,invert=invert)])


###########################################
#do some fitting for lightcurves stars........

#now fit a sin-function with the period to the lightcurve
def fitfunc_sin(time, p0, p1, p2, p3):
    z = p0 + p1 * numpy.sin( p2 + 2.0*math.pi*time/p3 )
        #alternatively we know that sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
        #hence maybe use this to do the fit as might help with removing issues with initial guess for phase
        #actually seems to work even less well than the straight fit!
        #a=p2
        #b=2.0*math.pi*time/p3
        #z = p0 + p1*numpy.sin(a)*numpy.cos(b) + p1*numpy.cos(a)*numpy.sin(b)
    return z

def fitfunc_sinother(time, p):
    z = p[0] + p[1] * numpy.sin( p[2] + 2.0*math.pi*time/p[3] )
    return z

def fitfunc_sin2(time, p0, p1, p2, p3):
    z = p0 + p1 * numpy.sin( p2 + 1.0*math.pi*time/p3 )**2.0
    return z

def fitfunc_abssin(time, p0, p1, p2, p3):
    z = p0 - p1 * numpy.abs(numpy.sin( p2 + 1.0*math.pi*time/p3 ))
    return z

def fitfunc_abssin2(time, p0, p1, p2, p3):
    z = p0 - p1 * numpy.abs(numpy.sin( p2 + 1.0*math.pi*time/p3 )**2.0)
    return z

def fitfunc_sin2(p,time):
        #z = p[0] - numpy.abs(p[1] * (numpy.cos( (p[2] + 2.0*math.pi*time/p[3]) )))**p[4]
    z = p[0] + p[1] * numpy.cos( p[2] + 2.0*math.pi*time/p[3] ) #+ p[4]
        #z = p[0] + p[1] * numpy.sin( (p[2] + 2.0*math.pi*time/p[3]) ) + p[4] * numpy.sin( (p[5] + 2.0*math.pi*time/p[6]) )
    return z

def errfunc_sin2(p,mag,time,magerr):
	#if ( (p[2] >= 0) and (p[2] <= 2.0*math.pi) and (p[1] >= 0.0)): #limits for the parameters!
    err =  numpy.abs((mag - fitfunc_sin2(p,time))/magerr)
        #else:
        #	err = 1000.
    return err

def fitfunc_linear2(uv,p0,p1,p2):
    z = p0 + p1 * uv[0] + p2 * uv[1]
    return z

def errfunc_linear2(p,mag,magerr,u,v):
    err =  numpy.abs((mag - fitfunc_linear2(p,u,v))/magerr)
    return err

#function to plot a periodogram
#returns on demand only fit parameters of last filter worked on! hence run for single filter if interested in all numbers
def periodogram(lightcurve_in, filt_list="", col_list="", name="",retparam="",fixperiod="",talk=""):
    if filt_list == "": filt_list = ["U", "B", "V", "R", "I", "HA"]
    if col_list == "": col_list = ['Purple','Blue','Green','Red','Black','Magenta']
    if name == "": name = 'test'
    if retparam == "": retparam='N'
    if talk == "": talk=0
    minperiod = 0.01	#minimum period to check
    maxperiod = 2.0	#minimum period to check
	
    for i in range(0,len(filt_list)):
        filt = filt_list[i]
        if (talk == 1):
        	print('working on filter ',filt)
        col = col_list[i]
        lightcurve = extract_data(lightcurve_in,prop_name='filter',id_list=filt,invert='')

    f = numpy.linspace(0.01, 10, 1000000) #checks 10 to 100 days
	#f = numpy.linspace(0.01, 100, 1000000) #checks 0.01 to 100 days
    lightcurve=sort_time(lightcurve)
	#pgram = signal.lombscargle(time, mag, f)
    pgram = LombScargle(lightcurve['date'], lightcurve['calibrated_magnitude']).power(f)
	#f,pgram = LombScargle(lightcurve['date'], lightcurve['calibrated_magnitude']).autopower()
	
	#find the maximum in periodogram at periods above minperiod
    maxper = numpy.argmax(pgram[numpy.where( 1.0/f > minperiod)])
        #either use a preset period of the peak one from the periodogram
    period = fixperiod
    if fixperiod == "": period = 1.0*1./f[maxper]
        ##################################
        #or if the fixed period is given as negative, estimate the period from the periodograms but only within 10% of the abs(period) given
        #this might allow to trace period changes better than the sine-fitting
    if (fixperiod < 0.0):
        check = numpy.where( (1.0/f > -0.9*fixperiod) & (1.0/f < -1.1*fixperiod) )
        f_cut = f[check[0]]
        pgram_cut = pgram[check[0]]
        maxper = numpy.argmax(pgram_cut)
        period = 1.0/f_cut[maxper]
                #print(period)
                #reset fixperiod to abs value to ensure plotting is working
        fixperiod = -fixperiod
                
        #period = 7.36447820994 #lkha146
        #period = 0.82457 #0.82457 #4.73414307661 #v1598cyg
        #period = period/(1.0/2.42)
    if (talk == 1):
        print("period: ", period)
	#print("time zero: ", numpy.min(lightcurve['date']))

	#plot the periodogram
    plt.clf()
    plt.figure(figsize=(6,4))
    plt.rcParams.update({'font.size': 8})
    rectangle = [0.1,0.55,0.35,0.35]
    ax = plt.axes(rectangle)
    plt.plot(1/f, pgram, 'k-', label='  P='+numpy.str(period)[:10]+'days')
    plt.xlim(0,2.0*period)
	#plt.xlim(0,6.0)
    plt.xlabel('Time [d]')
    plt.ylabel('frequency')
    plt.legend(loc='upper center', handletextpad=0.01)
	
	#plot the lightcurve
    rectangle = [0.55,0.55,0.35,0.35]
    ax = plt.axes(rectangle)
    plt.plot(lightcurve['date']-2400000.5, lightcurve['calibrated_magnitude'], color='0.7', marker='.', linestyle='solid', markersize=0.1)
    plt.plot(lightcurve['date']-2400000.5, lightcurve['calibrated_magnitude'], color=col,  marker='o', linestyle='none', markersize=1)
	#plt.ylim(numpy.max(lightcurve['calibrated_magnitude']) + 0.05,numpy.min(lightcurve['calibrated_magnitude'])-0.05)
    plt.ylim(numpy.median(lightcurve['calibrated_magnitude'])+5.0*numpy.std(lightcurve['calibrated_magnitude']),numpy.median(lightcurve['calibrated_magnitude'])-5.0*numpy.std(lightcurve['calibrated_magnitude']))
    pyplot.locator_params(axis='x', nbins=4)
    pyplot.locator_params(axis='y', nbins=5)
    plt.xlabel('Time, MJD [d]')
    plt.ylabel('Magnitude {} [mag]'.format(filt))
	
    mag = numpy.array(lightcurve['calibrated_magnitude'])
    magerr = numpy.array(lightcurve['calibrated_error'])
        
        #making guesses for parameters
    initial_offset = numpy.nanmedian(lightcurve['calibrated_magnitude'])
    initial_amplitude = 2*numpy.sqrt(2)*numpy.nanstd(lightcurve['calibrated_magnitude'])
    initial_period = period
    parameters = [initial_offset,initial_amplitude,1,initial_period]

        ######################################
        #make best guess for phase - not really working, hence abandoned!!!!
        #u = numpy.cos(2.0*math.pi*lightcurve['date']/initial_period)        
        #v = numpy.sin(2.0*math.pi*lightcurve['date']/initial_period)        
        #check=numpy.where(mag >= -99)
        #param_linear, success = curve_fit(fitfunc_linear2, [u[check[0]],v[check[0]]], mag[check[0]], sigma=magerr[check[0]], p0=[initial_offset,0,0],maxfev=1000000)
	#print('%%%%%%%%%%%%%%%%%%%%%%%%%')
        #initial_phase = numpy.mod(-math.pi/2+numpy.arctan2(param_linear[1],param_linear[2]),math.pi)
        #print('initial amplitude guess:', initial_amplitude)
        #print('best guess initial amplitude:', numpy.sqrt(param_linear[1]*param_linear[1]+param_linear[2]*param_linear[2]))
        #print('best guess initial phase: ',initial_phase)
        #####################################
	#parameters = [numpy.median(mag),0.0,0.0,1./period,0.0,0.0,400.]
	#param_cal, success = optimize.leastsq(errfunc_sin2, parameters, args=(mag,lightcurve['date'],magerr))
        
        #new version of fit using curvefit
        #param_cal, success = curve_fit(fitfunc_sin, lightcurve['date'], mag, sigma=magerr, p0=parameters, bounds=([numpy.median(lightcurve['calibrated_magnitude'])-1.0,0.0, 0.0, 0.9*period], [numpy.median(lightcurve['calibrated_magnitude'])+1.0, 1.0, 2.0*math.pi,1.1*period]))

	#try fitting by using sigma clipping and curvefit
    clip_sig = 3.0
    max_it = 30
    exit = 0
    iteration = 0
    check=numpy.where(mag >= -99)
    old_num = len(check[0])
    while ( (exit==0) and (iteration <= max_it)): #exit after max_it iterations or when no more points are removed
                #param_cal, success = curve_fit(fitfunc_sin, lightcurve['date'][check[0]], mag[check[0]], sigma=magerr[check[0]], p0=parameters,maxfev=10000, bounds=([numpy.nanmedian(lightcurve['calibrated_magnitude'][check[0]])-1.0,0.0, -10.0*math.pi, 0.9*period], [numpy.nanmedian(lightcurve['calibrated_magnitude'][check[0]])+1.0, 1.0, 10.0*math.pi,1.1*period]))
                #fit not working 100% as sometimes initial guess for the phase in particular makes fit not work
                #thus do several fits and median average results
                numiter = 11
                param_tries = numpy.zeros((4,numiter),dtype=float)
                for j in range(0,numiter):
                	#set the initial guess for the phase
                    parameters[2] = numpy.float(j) / numiter * 2.0 * math.pi
                        #then run the fit fith that
                    param_cal_res, success = curve_fit(fitfunc_sin, lightcurve['date'][check[0]], mag[check[0]], sigma=magerr[check[0]], p0=parameters,maxfev=100000000)
	                #set the amplitude to positive if negative
                    if (param_cal_res[1] < 0):
                        param_cal_res[1] = -param_cal_res[1]
		                #and hence the phase needs adding pi and renorming to 2pi max
                        if (talk == 1):
                            print(param_cal_res[2])
                        param_cal_res[2] = numpy.mod((param_cal_res[2] + math.pi), 2.0*math.pi)
                        #renorm all the phases to 2pi
                        param_cal_res[2] = numpy.mod(param_cal_res[2], 2.0*math.pi)
                        #store all the results
                        for k in range(0,4):
                                param_tries[k,j] = param_cal_res[k]
		#determine the median parameters
                #print(param_tries)
                param_cal = numpy.zeros((4),dtype=float)
                rms_array = numpy.zeros((numiter),dtype=float)
                ###############################
                #select only the top ?? of the fits in terms of rms
                for j in range(0,numiter):
	                rms_array[j] = numpy.std( mag[check[0]] -  fitfunc_sin(lightcurve['date'][check[0]],param_tries[0,j],param_tries[1,j],param_tries[2,j],param_tries[3,j]))
                ####################################################
                #as the best parameters pick the median of all the values
                #median all the offset, amplitudes and periods 
                #for k in (0,1,3):
                #	param_cal[k] = numpy.median(param_tries[k,:])
		#median all the phases
		#make components first
                #xlist = numpy.cos(param_tries[2])
                #ylist = numpy.sin(param_tries[2])
                #xmed = numpy.median(xlist)
                #ymed = numpy.median(ylist)
                #param_cal[2] = numpy.mod(numpy.arctan2(ymed, xmed),2.0*math.pi)
                ########################################################
                #alternatively pick the fit with the lowest rms instead of the median
                sorted = numpy.argsort(rms_array)
                for k in (0,1,2,3):
                	param_cal[k] = param_tries[k,sorted[0]]
                #print('Final parameters: ',param_cal)
                #determine the rms of the best fit
                rms = numpy.std( lightcurve['calibrated_magnitude'][check[0]] - fitfunc_sin(lightcurve['date'][check[0]],param_cal[0],param_cal[1],param_cal[2],param_cal[3]))
                #print(rms_array)
                #print(rms_array[sorted[0]])
                rms = rms_array[sorted[0]]
                check = numpy.where( numpy.abs(lightcurve['calibrated_magnitude'] - fitfunc_sin(lightcurve['date'],param_cal[0],param_cal[1],param_cal[2],param_cal[3])) <= clip_sig * rms) 
                new_num = len(check[0])
                if (new_num == old_num):	#check if additional points have been removed
                	exit = 1
                old_num = new_num
                parameters = param_cal #use output as new initial guess for next fit
                iteration = iteration + 1

	#print('fitted phase: ',param_cal[2])
	#print('%%%%%%%%%%%%%%%%%%%%%%%%%')
	#print('difference in phase: ',numpy.mod(numpy.abs(initial_phase - param_cal[2]),2.0*math.pi)*180.0/math.pi,'deg')
	#print('ratio of amplitudes: ',initial_amplitude / numpy.sqrt(param_linear[1]*param_linear[1]+param_linear[2]*param_linear[2]))
        #for checking
        #print(filt)
        #print(param_cal)
        #print(numpy.sqrt(numpy.diag(success)))
	#print('Fit RMS = ',rms,' mag')
	
    if (talk == 1):
        print('Fit parameters in ',filt)
        print('Median[mag], Amplitude[mag], Phase0, Period[days]')
        print(param_cal)
        print('Fit parameter uncertainties')
        print(numpy.sqrt(numpy.diag(success)))
        print('Fit RMS = ',rms,' mag')
    period = param_cal[3]
        
    phase = numpy.array(lightcurve['date']) * 0.0
    for i in range(0,len(lightcurve['date'])):
		#phase[i] = (lightcurve['date'][i] - numpy.min(lightcurve['date']))/period - numpy.floor((lightcurve['date'][i] - numpy.min(lightcurve['date']))/period) #+ ps
        phase[i] = numpy.mod( param_cal[2] + 2.0*math.pi*lightcurve['date'][i]/period, 2.0*math.pi) / (2.0*math.pi) #+ ps
        if (phase[i] >= 1):
            phase[i] = phase[i]-1.
        if (phase[i] <= 0):
            phase[i] = phase[i]+1.
	#determine phase at test date
    testdate=2458630.0
        #try new version of test date
        #testdate=0.5 * (numpy.max(lightcurve['date']) + numpy.min(lightcurve['date']))
    ph=numpy.mod( param_cal[2] + 2.0*math.pi*testdate/period, 2.0*math.pi) / (2.0*math.pi)
    if (ph >= 1):
        ph = ph-1.
    if (ph <= 0):
        ph = ph+1.
    if (talk == 1):
        print('Phase at 2458630 is: ',ph)


    check0 = numpy.where(fitfunc_sin(lightcurve['date'],param_cal[0],param_cal[1],param_cal[2],param_cal[3]) == numpy.max(fitfunc_sin(lightcurve['date'],param_cal[0],param_cal[1],param_cal[2],param_cal[3])))
    ps = phase[check0[0][0]] #the [0][0] makes sure it works even if there is more than one entry in the check list
    phase = phase + ps #phase shift for plot - to move maximum to phase=0,1

	#plot the folded light curve
    rectangle = [0.1,0.1,0.8,0.35]
    ax = plt.axes(rectangle)
    plt.plot(phase[check[0]]-1, lightcurve['calibrated_magnitude'][check[0]], color=col,  marker='o', linestyle='none', markersize=1)
    plt.plot(phase[check[0]], lightcurve['calibrated_magnitude'][check[0]], color=col,  marker='o', linestyle='none', markersize=1)
    plt.plot(phase[check[0]]+1, lightcurve['calibrated_magnitude'][check[0]], color=col,  marker='o', linestyle='none', markersize=1)
    plt.ylim(numpy.max(lightcurve['calibrated_magnitude'][check[0]]) + 0.05,numpy.min(lightcurve['calibrated_magnitude'][check[0]])-0.05)
    plt.xlim(0,2)
    plt.xlabel('Phase')
    plt.ylabel('Magnitude {} [mag]'.format(filt))
    plt.plot(phase[check[0]]-1,fitfunc_sin(lightcurve['date'][check[0]],param_cal[0],param_cal[1],param_cal[2],param_cal[3]),'ko', markersize=1)
    plt.plot(phase[check[0]],fitfunc_sin(lightcurve['date'][check[0]],param_cal[0],param_cal[1],param_cal[2],param_cal[3]),'ko', markersize=1)
    plt.plot(phase[check[0]]+1,fitfunc_sin(lightcurve['date'][check[0]],param_cal[0],param_cal[1],param_cal[2],param_cal[3]),'ko', markersize=1)

    plt.savefig(name+'_'+filt+'.pdf', format='pdf', bbox_inches='tight', dpi=600)
    plt.savefig(name+'_'+filt+'.png', format='png', bbox_inches='tight', dpi=600)
	#plt.show()
        
        #for testing - subtract the fit from the data, in case 2nd periodogram analysis is needed for 2nd frequency
        #lightcurve['calibrated_magnitude'] = lightcurve['calibrated_magnitude'] - fitfunc_sin2(param_cal,lightcurve['date']) + numpy.nanmedian(lightcurve['calibrated_magnitude'])
    plt.clf()
    plt.cla()
    plt.close()
    if (retparam == 'N'):
        return(lightcurve_in)
    if (retparam == 'Y'):
        return(lightcurve_in, param_cal, numpy.sqrt(numpy.diag(success)), ph, rms)


################### Function to truncate color map ###################
def truncate_colormap(cmapIn='seismic', minval=0.0, maxval=1.0, n=100):
    '''truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)'''    
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(numpy.linspace(minval, maxval, n)))

    return new_cmap

#cmap_mod = truncate_colormap(minval=.2, maxval=.8)  # calls function to truncate colormap
#function to make the fingerprint plots
def probplot(lc, filt_list=""):
    if filt_list == "": filt_list = ["U", "B", "V", "R", "I", "HA"]

    for filt in filt_list:
        lc_temp = extract_data(lc,prop_name='filter',id_list=filt,invert='')

        mag = lc_temp["calibrated_magnitude"]
        error = lc_temp["calibrated_error"]
        time = lc_temp["date"]

        sort_t = numpy.argsort(time)
        time = time[sort_t]
        mag = mag[sort_t]
        error = error[sort_t]

        print("Length of mag array is:", len(mag))

        counter = 0
        N = len(mag)
        size = int((N * (N - 1)) / 2)

        deltamag = numpy.zeros(size)
        deltatime = numpy.zeros(size)
        deltaerror = numpy.zeros(size)

        for i in range(0, N - 1):
            for j in range(i + 1, len(mag)):
                deltamag[counter] = mag[j] - mag[i]
                deltatime[counter] = time[j] - time[i]
                deltaerror[counter] = numpy.sqrt((error[j])**2 + (error[i])**2)
                counter += 1

        # Plots graphs of the change in magnitude vs change in time for each filter
        plt.plot(deltatime, deltamag, linestyle="none", marker=".", markersize=2)
        plt.xscale("log")
        plt.xlabel("$\Delta t$ (days)")
        plt.ylabel("$\Delta m_{}$ (mag)".format(filt))
        plt.suptitle("Change in {} magnitude vs time of {}".format(filt, lc_temp["name"][0]))
        plt.gca().invert_yaxis()
        #plt.show()
        # plt.savefig(os.getcwd() + "/corrections/images/magvtime_{}_{}.png".format(lc_temp["name"][0], filt), format="pdf", bbox_inches="tight", dpi=600)
        plt.clf()

        x_min = -1
        x_max = math.log10(numpy.max(time) - numpy.min(time))
        x_bin_num = 40  # 100  # 40
        #xbins = numpy.arange(x_min, x_max, float(abs(x_min) + abs(x_max)) / x_bin_num)
        xbins = numpy.linspace(x_min, x_max, x_bin_num)

        y_min = -1
        y_max = 1
        y_bin_num = 40
        #ybins = numpy.arange(y_min, y_max, float(abs(y_min) + abs(y_max)) / y_bin_num)
        ybins = numpy.linspace(y_min, y_max, y_bin_num)

        # Creating histogram from filtered data.
        timeerror = numpy.where((deltatime > 0) & (deltaerror < .3))
        H, xedges, yedges = numpy.histogram2d(numpy.log10(deltatime[timeerror[0]]), deltamag[timeerror[0]], bins=(xbins, ybins))

        # Normalising each column to 1.
        Hist_norm = numpy.zeros(H.shape)
        for c1, i in enumerate(H):
            for c2, j in enumerate(i):
                if numpy.sum(i):
                    Hist_norm[c1][c2] = j / numpy.sum(i)
                else:
                    Hist_norm[c1][c2] = 0  # numpy.nan

        # Plotting colour diagram for normalised histogram
        plt.pcolor(xedges, yedges, Hist_norm.T, cmap=plt.cm.viridis)
        plt.colorbar()
        plt.xlabel("$\Delta t$ $(log_{10}(days))$")
        plt.ylabel("$\Delta m_{}$ (mag)".format(filt))
        plt.suptitle("Intensity map showing change in {} magnitude vs time of {}".format(filt, lc_temp["name"][0]))
        plt.gca().invert_yaxis()
        plt.clim(0, .5)
        # plt.savefig(os.getcwd() + "/corrections/images/fingerprint_rough_{}_{}.png".format(lc_temp["name"][0], filt), format="pdf", bbox_inches="tight", dpi=600)
        #plt.show()
        plt.clf()

        # Plotting colour diagram for smoothed normalised histogram
        #plt.imshow(Hist_norm.T, interpolation="bicubic", cmap=plt.cm.viridis, extent=[x_min, x_max, y_max, y_min], aspect="auto")
        #plt.imshow(Hist_norm.T, interpolation="bicubic", cmap=plt.cm.seismic, extent=[x_min, x_max, y_max, y_min], aspect="auto")

        cmap_mod = truncate_colormap(cmapIn='viridis', minval=0.0, maxval=numpy.max(Hist_norm)/0.2, n=256)  # calls function to truncate colormap
        plt.imshow(Hist_norm.T, interpolation="bicubic", cmap=cmap_mod, extent=[x_min, x_max, y_max, y_min], aspect="auto")
        plt.xlabel("$\Delta t$ $(log_{10}(days))$")
        plt.ylabel("$\Delta m_{}$ (mag)".format(filt))
        plt.colorbar()
        #plt.colorbar(extend='both')
        #plt.clim(0, numpy.max(Hist_norm))
        # plt.savefig(os.getcwd() + "/corrections/images/fingerprint_smooth_{}_{}.png".format(lc_temp["name"][0], filt), format="pdf", bbox_inches="tight", dpi=600)
        plt.savefig('test_fingerprint.pdf', format='pdf', bbox_inches='tight', dpi=600)
        plt.savefig('test_fingerprint.png', format='png', bbox_inches='tight', dpi=600)
        #plt.show()
        plt.clf()
        plt.close()


#function to convert lightcurve into a python dta structure
def convert_to_datastructure(lightcurve,name):
	#define the datastructure
    datatype = numpy.dtype([('name',str,80),						#name of the object the lightcurve is associated with
        				     ('date',float),							#date/time (JD) of the observation
                                             ('calibrated_magnitude',float),		#calibrated apparent magnitude in the filter
                                             ('alpha_j2000',float),				#Ra (J2000) of the measurement
                                             ('delta_j2000',float),				#DEC (J2000) of the measurement
                                             ('calibrated_error',float),			#uncertainty of the calibrated apparent magnitude
                                             ('id',int),							#ID number of the star measurement
                                             ('filter',str,80),						#filter into which the photometry is calibrated into
                                             ('original_filter',str,80),			#original filter name used by observer
                                             ('x',float),							#x pixel position of source on the image
                                             ('y',float),							#y pixel position of source on the image
                                             ('med_mag',float),						#median magnitude in the filter used over timescale determined by median filter
                                             ('U',float),							#U band magnitude for this datapoint, determined by medmag_time
                                             ('B',float),							#B band magnitude for this datapoint, determined by medmag_time
                                             ('V',float),							#V band magnitude for this datapoint, determined by medmag_time
                                             ('R',float),							#R band magnitude for this datapoint, determined by medmag_time
                                             ('I',float),							#I band magnitude for this datapoint, determined by medmag_time
                                             ('HA',float),							#HA band magnitude for this datapoint, determined by medmag_time
                                             ('Ue',float),							#U band magnitude uncertainty for this datapoint, determined by medmag_time
                                             ('Be',float),							#B band magnitude uncertainty for this datapoint, determined by medmag_time
                                             ('Ve',float),							#V band magnitude uncertainty for this datapoint, determined by medmag_time
                                             ('Re',float),							#R band magnitude uncertainty for this datapoint, determined by medmag_time
                                             ('Ie',float),							#I band magnitude uncertainty for this datapoint, determined by medmag_time
                                             ('HAe',float),							#HA band magnitude uncertainty for this datapoint, determined by medmag_time
                                             ('dips',float),							#Is the datapoint part of a dip (-1), burst (+1), or not (0)
                                             ('dips_num',float),						#if dips=+-1 then indicate to which dip/burst it belongs
                                             ('magnitude_rms_error',float),		#rms of the calibrated_magnitudes in the same filter across the entire lightcurve
                                             ('fwhm_world',float),				#seeing, or FWHM of the star in degrees
                                             ('org_cal_mag', float),				#This is the original calibrated magnitude before colour correction
                                             ('org_cal_err', float),				#This is the original calibrated magnitude error before colour correction
                                             ('observation_id',int),				#ID number of the observation in the database
                                             ('user_id',int),						#ID number of the user that uploaded the image to the database
                                             ('target',int),						#ID number of the target field the source has been uploaded as
                                             ('flags',int),						#Flags from the SourceExtractor added during the photometry (additionally +512 if source too bright for calibration, +1024 if too faint for calibration))
                                             ('magnitude',float),				#uncalibrated magnitude of the star in the database
                                             ('device_id',int),						#ID number of the device the data was taken with 
                                             ('fits_id',int)])						#ID number of the FITS file the star has been measured on
        
    datastructure = numpy.empty(len(lightcurve),dtype=datatype)
	#fill the datastructure
    counter=0
    for star in lightcurve:
            datastructure[counter]['name']=name
            datastructure[counter]['date']=star['date']
            datastructure[counter]['calibrated_magnitude']=star['calibrated_magnitude']
            datastructure[counter]['alpha_j2000']=star['alpha_j2000']
            datastructure[counter]['delta_j2000']=star['delta_j2000']
            datastructure[counter]['calibrated_error']=star['calibrated_error']
            datastructure[counter]['id']=star['id']
            datastructure[counter]['filter']=star['filter']
            datastructure[counter]['original_filter']=star['original_filter']
            datastructure[counter]['x']=star['x']
            datastructure[counter]['y']=star['y']
            datastructure[counter]['med_mag']=-99.0
            datastructure[counter]['U']=-99.0
            datastructure[counter]['B']=-99.0
            datastructure[counter]['V']=-99.0
            datastructure[counter]['R']=-99.0
            datastructure[counter]['I']=-99.0
            datastructure[counter]['HA']=-99.0
            datastructure[counter]['Ue']=-99.0
            datastructure[counter]['Be']=-99.0
            datastructure[counter]['Ve']=-99.0
            datastructure[counter]['Re']=-99.0
            datastructure[counter]['Ie']=-99.0
            datastructure[counter]['HAe']=-99.0
            datastructure[counter]['dips']=0.0
            datastructure[counter]['dips_num']=0.0
            datastructure[counter]['magnitude_rms_error']=star['magnitude_rms_error']
            datastructure[counter]['org_cal_mag']=star['calibrated_magnitude']
            datastructure[counter]['org_cal_err']=star['calibrated_error']
            datastructure[counter]['fwhm_world']=star['fwhm_world']
            datastructure[counter]['observation_id']=star['observation_id']
            datastructure[counter]['user_id']=star['user_id']
            datastructure[counter]['target']=star['target']
            datastructure[counter]['flags']=int(float(star['flags']))
            datastructure[counter]['magnitude']=star['magnitude']
            datastructure[counter]['device_id']=star['device_id']
            datastructure[counter]['fits_id']=star['fits_id']
            counter=counter+1
        
    return(datastructure)

#function that extracts a single star lightcurve from what the sql querie returns - simplifying the structure
def extract_single_star(lightcurve):
    if (len(lightcurve) > 1):
        count=numpy.zeros(len(lightcurve['seperated']))
        print('there are ',len(lightcurve['seperated']),' stars in the search radius')
        i=0
        for cluster in lightcurve['seperated']:
            count[i]=int(len(lightcurve['seperated'][cluster]))
            i = i + 1
		#return just the single star lightcurve
        print('extracting number ',numpy.argmax(count)+1, ' which has', count[numpy.argmax(count)], ' entries.')
        return(lightcurve['seperated'][lightcurve['seperated'].keys()[numpy.argmax(count)]])
    if (len(lightcurve) == 1):
        return(lightcurve)

#takes a lightcurve and filter set and returns the number of images in each filter for a single star lightcurve
def check_filter_single(lightcurve_data, filt_list=''): 
	if filt_list=='': filt_list=['B','V','R','I','HA']
	filt_list=numpy.array(filt_list)
	counter = numpy.zeros(len(filt_list))
	for i in range(0,len(filt_list)):
		check = numpy.where(lightcurve_data['filter'] == filt_list[i])
		counter[i] = len(check[0])
	print('There are ',counter,' datapoints in the filters', filt_list)
	return(counter)


#takes a lightcurve and only returns the datapoints which are in a supplied list of filters (e.g. only one filter)
def extract_filter(lightcurve_data, filt_list=''):
    if filt_list=='': filt_list=['B','V','R','I','HA']
    filt_list=numpy.array(filt_list)
    counter = 0
        #print('#################################')
	#print('# extracting filters ', filt_list, ' from light curve...')
    for i in range(len(lightcurve_data),0,-1):
        check = numpy.where(filt_list == lightcurve_data[i-1]['filter'])
        if (len(check[0]) == 0):
            lightcurve_data = numpy.delete(lightcurve_data, i-1)
        #print('#################################')
    return(lightcurve_data)


def make_query(incoords,inradius):

    USER = 'sps_student'
    PASSWORD = 'sps_student'

	# Connect to the database
    try:
        print('Attempting to speak to server...')
        connection = pymysql.connect(host='129.12.24.29',
                                     user=USER,
                                     password=PASSWORD,
                                     db='imageportal',
                                     charset='utf8mb4',
                                     cursorclass=pymysql.cursors.DictCursor)
    except:
        raise ServerTalkException('...Failed to contact server!')

    print('...Connection established!')
    
    coords = SkyCoord(incoords, frame='fk5', unit=u.degree)
	# Covert whatever we have got to FK5
    coords = coords.transform_to('fk5')
    radius = inradius
	#make some changes should values by at the limits
    if radius:
        radius = radius
    else:
        radius = 5
    if coords.dec.degree == 90:
        dec = 89.99999
    else:
        dec = coords.dec.degree

    lightcurve_data, user_choices = index_stars(coords, radius, dec, connection)

	#Open a file for use in the next program and save the filter, flag, device id, error, userid
    f= open("test.txt","w+")
    for cluster in lightcurve_data['seperated']:
        for star in lightcurve_data['seperated'][cluster]:
			#for each element that is extracted from the server, write filer, mag, error and date
            f.write("{0[filter]}\t{0[flags]}\t{0[device_id]}\t{0[calibrated_magnitude]}\t{0[calibrated_error]}\t{0[date]}\t{0[user_id]}\n".format(star))
    f.close()

    return(lightcurve_data)



def catalogue_create(target_id):
    """Creates a catalogue of non-variable stars for a given target field."""
    def stetson(mag, sigma_mag):
        """
        Computes the stetson index for a given list of magnitudes and magnitude errors.
        If s is less than 0.1, the stars are considered non-variable.
        """

        def delta(mag, sigma_m, mean_m, num):
            """Computes delta for the stetson index."""
            delta_d = numpy.sqrt(num / (num - 1)) * (mag - mean_m) / sigma_m
            return delta_d

        num = len(mag)
        if num < 2:
            return 0

        mean_mag = numpy.nanmean(mag)
        d_mag = delta(mag, numpy.sqrt(sigma_mag), mean_mag, num)
        p_i = numpy.array([(d_mag**2) - 1])
        stetson_s = numpy.sum(numpy.sign(p_i) * numpy.sqrt(numpy.abs(p_i))) / (num * 1) + 1
        return stetson_s

    USER = "sps_student"
    PASSWORD = "sps_student"

    # Connect to the database
    try:
        print("Attempting to speak to server...")
        CONNECTION = pymysql.connect(host="129.12.24.29",
                                     user=USER,
                                     password=PASSWORD,
                                     db="imageportal",
                                     charset="utf8mb4",
                                     cursorclass=pymysql.cursors.DictCursor)
    except:
        raise ServerTalkException("...Failed to contact server!!")

    print("Creating catalogue for target_id {}".format(target_id))
    print("Retriving calibration dates...")
    jdates = {}
    for filt in "VRI":
        with CONNECTION.cursor() as cursor:
            sql3 = "SELECT t1.id, date, count(*) AS num \
            FROM observations t1, photometry t2 \
            WHERE t1.id = observation_id AND (target_id = %s) AND (filter = '{}') AND t2.flags <= 4 \
            GROUP BY t1.id ORDER BY num DESC LIMIT 1;".format(filt)
            cursor.execute(sql3, (target_id))
            calib_date = cursor.fetchone()
            jdates["{}".format(filt)] = calib_date["date"]
            print("Retrived date of {} for {}".format(jdates["{}".format(filt)], filt))

    jdates = {filtr: date for filtr, date in jdates.items() if date is not None}

    X = {}

    for filtr in jdates:
        print("Collecting data for the filter: {}".format(filtr))
        X[filtr] = {}
        with CONNECTION.cursor() as cursor:
            sql = "SELECT alpha_j2000, delta_j2000, flags \
            FROM photometry INNER JOIN observations \
            ON (photometry.observation_id = observations.id) \
            WHERE (date = %s) AND (target_id = %s) AND (filter = %s);"
            cursor.execute(sql, (jdates[filtr], target_id, filtr))
            X[filtr]["df"] = pd.DataFrame(cursor.fetchall())

        X[filtr]["df"] = X[filtr]["df"][(X[filtr]["df"]["flags"].astype(float) <= 4)]
        X[filtr]["df"] = X[filtr]["df"].reset_index(drop=True)

        X[filtr]["coords"] = SkyCoord(X[filtr]["df"]["alpha_j2000"].astype(float), X[filtr]["df"]["delta_j2000"].astype(float), frame="icrs", unit=u.degree)

    short_filt = min(X, key=lambda key: len(X[key]["coords"]))
    matches = {}
    for filtr in jdates:
        if filtr != short_filt:
            matches[filtr] = match_coordinates_sky(X[short_filt]["coords"], X[filtr]["coords"], nthneighbor=1)

    all_matches = [True] * len(X[short_filt]["coords"])
    for key, value in iter(matches.items()):
        all_matches = [i and (j <= 3.0 * u.arcsec) for i, j in list(zip(all_matches, value[1]))]  # value[1] is d2d

    matches[short_filt] = [i for i in range(0, len(all_matches))], [i for i in range(0, len(all_matches))], [i for i in range(0, len(all_matches))]

    for key, value in iter(matches.items()):
        X[key]["df"] = X[key]["df"].iloc[numpy.array(value[0])[numpy.array(all_matches)]]  # value[0] is idx
    for filtr in jdates:
        X[filtr]["df"] = X[filtr]["df"].reset_index(drop=True)

    cat = pd.DataFrame({"alpha_j2000": numpy.empty(len(X[short_filt]["df"]["alpha_j2000"]))*numpy.nan, "delta_j2000": numpy.empty(len(X[short_filt]["df"]["alpha_j2000"]))*numpy.nan})

    for filtr in "UBVRI":
        cat["{}_mag".format(filtr)] = numpy.nan
        cat["{}_error".format(filtr)] = numpy.nan
        if filtr in "VRI":
            cat["s{}".format(filtr)] = numpy.nan

    for count, coords in enumerate(list(zip(X[short_filt]["df"]["alpha_j2000"], X[short_filt]["df"]["delta_j2000"]))):
        print("Checking star {} of {} calibration stars.".format(count, (len(X[short_filt]["df"]["alpha_j2000"]) - 1)))
        lc = make_query("{} {}".format(coords[0], coords[1]), 3.0)

        lc1 = extract_single_star(lc)

        lc2 = convert_to_datastructure(lc1, "{}".format(coords[0]))

        df = pd.DataFrame(lc2)

        df = df[((df["calibrated_magnitude"] != 0) &
                 (df["calibrated_error"] != 0) &
                 (df["flags"].astype(float) <= 4))]

        temp_dfs = {}
        stetson_temp = {}

        for filtr in df['filter'].unique():
            temp_dfs[filtr] = df[df["filter"] == filtr]

        if (len(temp_dfs["V"]) >= 100) and (len(temp_dfs["R"]) >= 100) and (len(temp_dfs["I"]) >= 100): #use 100 datapoints, except sig-ori there 50
            for filt, dframe in iter(temp_dfs.items()):
                if filt in "VRI":
                    stetson_temp[filt] = stetson(dframe["calibrated_magnitude"].astype(float).values, dframe["calibrated_error"].astype(float).values)

            if max(stetson_temp.values()) <= 0.1: # usually 0.1, use 0.5: for gaia19fct as object very red and 0.1 removes all red stars, and sig-ori
                cat.loc[count, "alpha_j2000"] = coords[0]
                cat.loc[count, "delta_j2000"] = coords[1]
                for key, value in stetson_temp.items():
                    print('stetson key', key)
                    cat.loc[count, "s{}".format(key)] = value
                for key, value in temp_dfs.items():
                    print('mag key', key)
                    cat.loc[count, "{}_mag".format(key)] = numpy.nanmedian(temp_dfs[key]["calibrated_magnitude"])
                    cat.loc[count, "{}_error".format(key)] = numpy.nanmedian(temp_dfs[key]["calibrated_error"])
            else:
                print("Star too variable.")
                cat.drop([count], axis=0, inplace=True)
        else:
            print("Not enough data points.")
            cat.drop([count], axis=0, inplace=True)

    cat = cat.reset_index(drop=True)
    ordered_filts = [i for i in "VRI" if i in jdates.keys()]

    # Determines colour values for each star, e.g. V-I or B-V
    for i, j in enumerate(ordered_filts):
        for l in ordered_filts[i + 1:]:
            cat[j + "-" + l] = [cat[j + "_mag"][m] - cat[l + "_mag"][m] for m, n in enumerate(cat["alpha_j2000"])]

    if not os.path.exists(os.getcwd() + "/corrections/catalogues"):
        os.makedirs(os.getcwd() + "/corrections/catalogues")

    cat.to_csv(os.getcwd() + "/corrections/catalogues/{}_calib_cat.csv".format(target_id), index=True)

    print("...catalogue created!")


def correction(fits_id, star_coord, cat_coords, cat, star_lc):
    """Actually performs the correction."""
    ##############################
    #defining the fit function for least square optimization
    def colourfun(p,mag,col):
        z = p[0] + p[1]*col + p[2]*col*col + p[3]*mag + p[4]*mag*mag
        return z
    #define error function for least square optimization
    def colourerr(p,mag,col,offsets):
        #weight the fit by the magnitude - brighter objects get heigher weight
        weight = (mag-(numpy.nanmin(mag)-2.0))**2.0
        err = (offsets - colourfun(p,mag,col))/weight
        return err
    
    def rms(y):
        """Returns the root mean square value of a list of values """
        #return numpy.sqrt(numpy.mean([i**2 for i in y]))
        return numpy.std(y)

    #lcV = numpy.nanmedian(star_lc["calibrated_magnitude"][star_lc["filter"] == "V"])
    #print("lcV: {}".format(lcV))
    #lcI = numpy.nanmedian(star_lc["calibrated_magnitude"][star_lc["filter"] == "I"])
    #print("lcI: {}".format(lcI))
    #star_colour = lcV - lcI
    star_mag = numpy.nanmedian(star_lc["calibrated_magnitude"][star_lc["fits_id"] == fits_id])
    star_field = star_lc["target"][star_lc["fits_id"] == fits_id].values[0]
    star_user = star_lc["user_id"][star_lc["fits_id"] == fits_id].values[0]
    star_device = star_lc["device_id"][star_lc["fits_id"] == fits_id].values[0]
    star_filter = star_lc["filter"][star_lc["fits_id"] == fits_id].values[0]
    ####################################################
    #find the colour of the star from measurements within XXX days 
    XXX=5.0
    factor = 2.0
    limit = 1000.0
    star_date = star_lc["date"][star_lc["fits_id"] == fits_id].values[0]
    while (XXX < limit):
        
        if ( (len(star_lc["calibrated_magnitude"][ (numpy.abs(star_lc["date"] - star_date) <= XXX) & (star_lc["filter"] == "V")]) > 3) and (len(star_lc["calibrated_magnitude"][ (numpy.abs(star_lc["date"] - star_date) <= XXX) & (star_lc["filter"] == "I")]) > 3)):
            lcV = numpy.nanmedian(star_lc["calibrated_magnitude"][ (numpy.abs(star_lc["date"] - star_date) <= XXX) & (star_lc["filter"] == "V")])
            lcI = numpy.nanmedian(star_lc["calibrated_magnitude"][ (numpy.abs(star_lc["date"] - star_date) <= XXX) & (star_lc["filter"] == "I")])
            star_colour = lcV - lcI
            if ( (star_colour > -5.0) & (star_colour < 10.0)):
                XXX = XXX + limit
        XXX = XXX * factor
    print('Colour, magnitude, date, field, user, device, filter')
    print("star properties: {}".format(star_colour),star_mag, star_date, star_field, star_user, star_device, star_filter)
    #####################################################################################

    if not os.path.exists(os.getcwd() + "/corrections/corr_params/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/"):
        os.makedirs(os.getcwd() + "/corrections/corr_params/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/")

    if os.path.isfile(os.getcwd() + "/corrections/corr_params/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_corr_params.txt".format(fits_id)):
        print("trying to use file")
        with open(os.getcwd() + "/corrections/corr_params/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_corr_params.txt".format(fits_id), "r") as f:
            f = f.read().split("\n")

        corr_params = [float(i) for i in f[0].strip().split(" ")]  
        print('Corr Parameters: ',corr_params)
        err = float(f[1])
        if (err >= 90):
            #do not run correction as file is basically blanks
            star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "flags"] += 2048  # Gives us a way to filter out non colour calibrated data.
            star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_magnitude"] = numpy.nan
            star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_error"] = numpy.nan
            return star_lc 

        star_offset = colourfun(corr_params, star_mag, star_colour)
        print("Offset: {}".format(star_offset))
        print("Corrected mag: {} +/- {} from {}".format((float(star_mag) + star_offset), float(err), float(star_mag)))
        star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_magnitude"] = float(star_mag) + float(star_offset)
        star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_error"] = float(err)
        return star_lc

    else:

        USER = 'sps_student'
        PASSWORD = 'sps_student'
        	# Connect to the database
        try:
        	print('Attempting to speak to server...')
        	connection = pymysql.connect(host='129.12.24.29',
        		user=USER,
        		password=PASSWORD,
        		db='imageportal',
        		charset='utf8mb4',
        		cursorclass=pymysql.cursors.DictCursor)
        except:
        	raise ServerTalkException('...Failed to contact server!')
        print('...Connection established!')
       
    
        with connection.cursor() as cursor:
            sql = "SELECT * FROM photometry INNER JOIN observations \
            ON (photometry.observation_id = observations.id) \
            WHERE (fits_id = %s) AND (flags <= 4);"
            cursor.execute(sql, (fits_id))
            data = cursor.fetchall()
    
        data = pd.DataFrame(data)
    
        data = data[(data["calibrated_magnitude"].astype(float) != 0) &
                    (data["calibrated_magnitude"].astype(float) != -99.) &
                    (data["calibrated_error"].astype(float) != 0) &
                    (data["calibrated_error"].astype(float) != -99.)]
    
        data = data.reset_index(drop=True)
    
        filt = data["filter"][0]
        data_coords = SkyCoord(data["alpha_j2000"], data["delta_j2000"], frame="icrs", unit=u.degree)

        cat_matches = pd.DataFrame()
        data_matches = pd.DataFrame()
        idx, d2d, d3d = match_coordinates_sky(cat_coords, data_coords)

        def dup_matches(seq, val):
            start = -1
            locs = []
            while True:
                try:
                    loc = seq.index(val, start + 1)
                except ValueError:
                    break
                else:
                    locs.append(loc)
                    start = loc
            return locs

        list_idx = list(idx)
        list_d2d = list(d2d.arcsec)
        for count, ind in enumerate(list_idx):
            matching_idx = dup_matches(list_idx, ind)
            d_seps = [list_d2d[l] for k, l in enumerate(matching_idx)]
            d_sep = min(d_seps)
            if d_sep <= 3.:
                cat_matches = cat_matches.append(cat.iloc[[list_d2d.index(d_sep)]])
                data_matches = data_matches.append(data.iloc[[ind]])
        #print(cat_matches,data_matches)

        cat_matches.drop_duplicates(inplace=True)
        cat_matches = cat_matches.reset_index(drop=True)

        data_matches.drop_duplicates(inplace=True)
        data_matches = data_matches.reset_index(drop=True)
        data_matches["calibrated_magnitude"] = data_matches["calibrated_magnitude"].astype(float)



        print("There are {} of {} calibration stars in this fits file\n".format(len(cat_matches), len(cat_coords)))
        if len(cat_matches) < 10:
            #overwrite the magnitudes and errors and flags
            star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "flags"] += 2048  # Gives us a way to filter out non colour calibrated data.
            star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_magnitude"] = numpy.nan
            star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_error"] = numpy.nan
            #write a 'blank' correction file with zeros
            corr_params = [0.0,0.0,0.0,0.0,0.0]
            err = 99.0
            with open(os.getcwd() + "/corrections/corr_params/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_corr_params.txt".format(fits_id), "w") as f:
                for item in corr_params:
                    f.write("%s " % item)
                f.write("\n%s" % err)

            return star_lc

        data_matches["mag_offsets"] = numpy.nan  # Placeholders
        for count, mag in enumerate(list(zip(data_matches["calibrated_magnitude"].astype(float), cat_matches["{}_mag".format(filt)].astype(float)))):
            mag_offset = mag[1] - mag[0]
            data_matches.loc[count, "mag_offsets"] = mag_offset  # Replaces nans above.

        c_cols = cat_matches["V-I"]
        d_mags = data_matches["calibrated_magnitude"]
        d_offs = data_matches["mag_offsets"]
        c_mags = cat_matches["{}_mag".format(filt)].values
        
        print(len(c_cols),len(d_mags))
        #for i in range(0,len(c_cols)):
        #	print(i,d_mags[i],c_cols[i])

        # ----- [0, 0] (Top left plot) --------------------------------------------------------------- #
        corr_params = [0.0,0.0,0.0,0.0,0.0]
        popt0, pcov0 = optimize.leastsq(colourerr, corr_params, args=(d_mags,c_cols,d_offs))
        
        av_mag = float(numpy.nanmedian(d_mags))
        y_fit = numpy.arange(-1.0 , 5.0 , 0.1)
        xav = colourfun(popt0,av_mag,y_fit)

        # -------------------------------------------------------------------------------------------- #

        # ----- [1, 0] (Bottom left plot) ------------------------------------------------------------ #
        av_colour = float(numpy.nanmedian(c_cols))
        x_fit = numpy.arange(10, 19, 0.1)
        yav = colourfun(popt0,x_fit,av_colour)
        rms1 = rms(d_offs)

        # -------------------------------------------------------------------------------------------- #

        # ----- [1, 1] (Bottom right plot) ----------------------------------------------------------- #
        x11 = d_mags + colourfun(popt0,d_mags,c_cols)
        y11 = c_mags - x11
        rms11 = rms(y11)

        mask_2sig = stats.sigma_clip(y11, sigma=2, iters=None)
        c_2 = cat_matches[mask_2sig.mask]
        d_2 = data_matches[mask_2sig.mask]
        c_2sig = cat_matches[~mask_2sig.mask]
        d_2sig = data_matches[~mask_2sig.mask]
        c2_cols = c_2sig["V-I"]
        c2_mags = c_2sig["{}_mag".format(filt)].values
        d2_mags = d_2sig["calibrated_magnitude"]
        popt2sig, pcov2sig = optimize.leastsq(colourerr, corr_params, args=(d2_mags,c2_cols,c2_mags - d2_mags))
        d2_off = d2_mags + colourfun(popt2sig,d2_mags,c2_cols)

        mask_3sig = stats.sigma_clip(y11, sigma=3, iters=None)
        c_3 = cat_matches[mask_3sig.mask]
        d_3 = data_matches[mask_3sig.mask]
        c_3sig = cat_matches[~mask_3sig.mask]
        d_3sig = data_matches[~mask_3sig.mask]
        c3_cols = c_3sig["V-I"]
        c3_mags = c_3sig["{}_mag".format(filt)].values
        d3_mags = d_3sig["calibrated_magnitude"]
        popt3sig, pcov3sig = optimize.leastsq(colourerr, corr_params, args=(d3_mags,c3_cols,c3_mags - d3_mags))
        d3_off = d3_mags + colourfun(popt3sig,d3_mags,c3_cols)

        rms_fin = rms(c3_mags - (d3_off))
        rms_fin2 = rms(c2_mags - (d2_off))

        # -------------------------------------------------------------------------------------------- #

        # ----- [0, 1] (Top right plot) -------------------------------------------------------------- #

        # -------------------------------------------------------------------------------------------- #

        # plt.show()

        if not os.path.exists(os.getcwd() + "/corrections/images/pdfs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)):
            os.makedirs(os.getcwd() + "/corrections/images/pdfs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter))
        if not os.path.exists(os.getcwd() + "/corrections/images/pngs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)):
            os.makedirs(os.getcwd() + "/corrections/images/pngs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter))


        ######################################################################
        #making the two plots - re-written to use matplotlib.pylab instead of plyplot
        ######################################################################
        #get the min/max values of all the axis
        #min_mag = 10.0
        min_mag = numpy.nanmin(d_mags) - 0.25
        #max_mag = 19.0
        max_mag = numpy.nanmax(d_mags) + 0.25
        #min_col = -0.5
        min_col = numpy.nanmin(c_cols) - 0.1
        #max_col = 3.0
        max_col = numpy.nanmax(c_cols) + 0.1
        #min_off = -0.3
        min_off = numpy.nanmin(d_offs) - 0.1
        #max_off = 0.3
        max_off = numpy.nanmax(d_offs) + 0.1
        

        #define the first set of plots:
        rectangle1 = [0.08,0.55, 0.35,0.35]
        rectangle2 = [0.08,0.08, 0.35,0.35]
        rectangle3 = [0.55,0.55, 0.35,0.35]
        rectangle4 = [0.55,0.08, 0.35,0.35]
        axa1 = plt.axes(rectangle1)
        axa2 = plt.axes(rectangle2)
        axa3 = plt.axes(rectangle3)
        axa4 = plt.axes(rectangle4)
        axa1.set_xlabel("Calibration star colour [V-I]")
        axa1.set_ylabel("Magnitude Offset [mag]")
        axa2.set_xlabel("Calibration star magnitude [mag]")
        axa2.set_ylabel("Magnitude Offset [mag]")
        axa3.set_xlabel("Calibration star colour [V-I]")
        axa3.set_ylabel("Correction Offset [mag]")
        axa4.set_xlabel("Corrected calibration star magnitude [mag]")
        axa4.set_ylabel("Correction Offset [mag]")
        axa1.set_xlim((min_col,max_col))
        axa2.set_xlim((min_mag,max_mag))
        axa3.set_xlim((min_col,max_col))
        axa4.set_xlim((min_mag,max_mag))
        axa1.set_ylim((min_off,max_off))
        axa2.set_ylim((min_off,max_off))
        axa3.set_ylim((min_off,max_off))
        axa4.set_ylim((min_off,max_off))

        #plot the top left plot
        axa1.scatter(x=c_cols, y=d_offs, s=5, c="b", alpha=0.75)
        axa1.plot(y_fit, xav, c="k", linestyle="--", alpha=0.75)
        #plot the bottom left plot
        axa2.scatter(x=d_mags, y=d_offs, s=5, c="b", alpha=0.75)
        axa2.plot(x_fit, yav, c="k", linestyle="--", alpha=0.75)
        axa2.text(x=min_mag+0.5, y=min_off+0.05, s="RMS: {:.3f}".format(rms1), fontsize=5)
        #plot the bottom right plot
        axa4.scatter(x=d_mags + colourfun(popt0,d_mags,c_cols), s=5, y=y11, c="g", alpha=0.45)
        axa4.scatter(x=d_2["calibrated_magnitude"] + colourfun(popt0,d_2["calibrated_magnitude"],c_2["V-I"]), y=c_2["{}_mag".format(filt)].values - (d_2["calibrated_magnitude"] + colourfun(popt0,d_2["calibrated_magnitude"],c_2["V-I"])), s=5, c="y", alpha=1)#0.55)
        axa4.scatter(x=d_3["calibrated_magnitude"] + colourfun(popt0,d_3["calibrated_magnitude"],c_3["V-I"]), y=c_3["{}_mag".format(filt)].values - (d_3["calibrated_magnitude"] + colourfun(popt0,d_3["calibrated_magnitude"],c_3["V-I"])), s=5, c="r", alpha=0.75)
        axa4.text(x=min_mag+0.5, y=min_off+0.05, s="RMS: {:.3f}, 3sig: {:.3f}".format(rms11, rms(c3_mags - (d3_mags + colourfun(popt0,d3_mags,c3_cols)))), fontsize=5)
        #plot the top right plot
        axa3.scatter(x=c_cols, y=y11, s=5, c="g", alpha=0.5)
        #horizontal lines
        axa1.axhline(y=0, alpha=0.5, color="k", linestyle=":")
        axa2.axhline(y=0, alpha=0.5, color="k", linestyle=":")
        axa3.axhline(y=0, alpha=0.5, color="k", linestyle=":")
        axa4.axhline(y=0, alpha=0.5, color="k", linestyle=":")

	#plt.show()
        #save the plot
        plt.savefig(os.getcwd() + "/corrections/images/pdfs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_{}_{}_initial.pdf".format(fits_id, filt, data["user_id"][0]), format="pdf", bbox_inches="tight", dpi=600)
        plt.savefig(os.getcwd() + "/corrections/images/pngs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_{}_{}_initial.png".format(fits_id, filt, data["user_id"][0]), format="png", bbox_inches="tight", dpi=600)

        #clear the plot
        plt.clf()

        #define the second plot
        axb1 = plt.axes(rectangle1)
        axb2 = plt.axes(rectangle2)
        axb3 = plt.axes(rectangle3)
        axb4 = plt.axes(rectangle4)
        axb1.set_xlabel("Calibration star colour [V-I]")
        axb1.set_ylabel("Magnitude Offset [mag]")
        axb2.set_xlabel("Calibration star magnitude [mag]")
        axb2.set_ylabel("Magnitude Offset [mag]")
        axb3.set_xlabel("Calibration star colour [V-I]")
        axb3.set_ylabel("Correction Offset [mag]")
        axb4.set_xlabel("Corrected calibration star magnitude [mag]")
        axb4.set_ylabel("Correction Offset [mag]")
        axb1.set_xlim((min_col,max_col))
        axb2.set_xlim((min_mag,max_mag))
        axb3.set_xlim((min_col,max_col))
        axb4.set_xlim((min_mag,max_mag))
        axb1.set_ylim((min_off,max_off))
        axb2.set_ylim((min_off,max_off))
        axb3.set_ylim((min_off,max_off))
        axb4.set_ylim((min_off,max_off))

        #plot the top left plot
        axb1.scatter(x=c_cols, y=d_offs, s=5, c="b", alpha=0.75)  # Top Left plot 2
        axb1.plot(y_fit, xav, c="k", linestyle="--", alpha=0.75)
        #plot the bottom left plot
        axb2.scatter(x=d_mags, y=d_offs, s=5, c="b", alpha=0.75)
        axb2.plot(x_fit, yav, c="k", linestyle="--", alpha=0.75)
        axb2.text(x=min_mag+0.5, y=min_off+0.05, s="RMS: {:.3f}".format(rms1), fontsize=5)
        #plot the bottom right plot
        axb4.scatter(x=d3_mags + colourfun(popt3sig,d3_mags,c3_cols), y=c3_mags - (d3_mags + colourfun(popt3sig,d3_mags,c3_cols)), s=5, c="g", alpha=0.5)
        axb4.text(x=min_mag+0.5, y=min_off+0.05, s="RMS: {:.3f}".format(rms(c3_mags - (d3_off))), fontsize=5)
        axb4.axhline(y=rms_fin, c="k", linestyle="--", alpha=0.75)
        axb4.axhline(y=-rms_fin, c="k", linestyle="--", alpha=0.75)
        #plot the top right plot
        axb3.scatter(x=c3_cols, y=c3_mags - (d3_off), s=5, c="g", alpha=0.5)
        axb3.axhline(y=rms_fin, c="k", linestyle="--", alpha=0.75)
        axb3.axhline(y=-rms_fin, c="k", linestyle="--", alpha=0.75)
        #horizontal lines
        axb1.axhline(y=0, alpha=0.5, color="k", linestyle=":")
        axb2.axhline(y=0, alpha=0.5, color="k", linestyle=":")
        axb3.axhline(y=0, alpha=0.5, color="k", linestyle=":")
        axb4.axhline(y=0, alpha=0.5, color="k", linestyle=":")

	#plt.show()
        #save the plot
        plt.savefig(os.getcwd() + "/corrections/images/pdfs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_{}_{}_final.pdf".format(fits_id, filt, data["user_id"][0]), format="pdf", bbox_inches="tight", dpi=600)
        plt.savefig(os.getcwd() + "/corrections/images/pngs/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_{}_{}_final.png".format(fits_id, filt, data["user_id"][0]), format="png", bbox_inches="tight", dpi=600)

        #clear the plot
        plt.clf()

        err = rms(c3_mags - d3_off)

        with open(os.getcwd() + "/corrections/corr_params/"+str(star_field)+"/"+str(star_user)+"/"+str(star_device)+"/"+str(star_filter)+"/{}_corr_params.txt".format(fits_id), "w") as f:
            for item in popt3sig:
                f.write("%s " % item)
            f.write("\n%s" % err)

        star_offset = colourfun(popt3sig,star_mag,star_colour)

        print("Offset: {}".format(star_offset))
        print("Corrected mag: {} +/- {} from {}".format((float(star_mag) + star_offset), float(err), float(star_mag)))

        star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_magnitude"] = float(star_mag) + float(star_offset)
        star_lc.at[star_lc[star_lc["fits_id"] == fits_id].index[0], "calibrated_error"] = float(err)
        return star_lc


def colour_correction(lightcurve):
    """Colour correction for stars' lightcurve."""
    USER = "sps_student"
    PASSWORD = "sps_student"
    ra = float(numpy.median(lightcurve['alpha_j2000']))
    dec = float(numpy.median(lightcurve['delta_j2000']))
    # Connect to the database
    try:
        print("Attempting to speak to server...")
        CONNECTION = pymysql.connect(host="129.12.24.29",
                                     user=USER,
                                     password=PASSWORD,
                                     db="imageportal",
                                     charset="utf8mb4",
                                     cursorclass=pymysql.cursors.DictCursor)
    except:
        raise ServerTalkException("...Failed to contact server!!")

    datatype = numpy.dtype([("name", str, 80),			        # Name of the object the lightcurve is associated with
                            ("date", float),			        # Date/time (JD) of the observation
                            ("calibrated_magnitude", float),	# Calibrated apparent magnitude in the filter
                            ("alpha_j2000", float),			    # Ra (J2000) of the measurement
                            ("delta_j2000", float),			    # DEC (J2000) of the measurement
                            ("calibrated_error", float),		# Uncertainty of the calibrated apparent magnitude
                            ("id", int),				        # ID number of the star measurement
                            ("filter", str, 80),			    # Filter into which the photometry is calibrated into
                            ("original_filter", str, 80),		# Original filter name used by observer
                            ("x", float),				        # X pixel position of source on the image
                            ("y", float),				        # Y pixel position of source on the image
                            ("med_mag", float),			        # Median magnitude in the filter used over timescale
                            ("U", float),				        # U band magnitude for this datapoint, determined by medmag_time
                            ("B", float),				        # B band magnitude for this datapoint, determined by medmag_time
                            ("V", float),				        # V band magnitude for this datapoint, determined by medmag_time
                            ("R", float),				        # R band magnitude for this datapoint, determined by medmag_time
                            ("I", float),				        # I band magnitude for this datapoint, determined by medmag_time
                            ("HA", float),				        # HA band magnitude for this datapoint, determined by medmag_time
                            ("Ue", float),				        # U band magnitude uncertainty for this datapoint, determined by medmag_time
                            ("Be", float),				        # B band magnitude uncertainty for this datapoint, determined by medmag_time
                            ("Ve", float),				        # V band magnitude uncertainty for this datapoint, determined by medmag_time
                            ("Re", float),				        # R band magnitude uncertainty for this datapoint, determined by medmag_time
                            ("Ie", float),				        # I band magnitude uncertainty for this datapoint, determined by medmag_time
                            ("HAe", float),				        # HA band magnitude uncertainty for this datapoint, determined by medmag_time
                            ("dips", float),			        # Is the datapoint part of a dip (-1), burst (+1), or not (0)
                            ("dips_num", float),			    # If dips=+-1 then indicate to which dip/burst it belongs
                            ("magnitude_rms_error", float),		# Rms of the calibrated_magnitudes in the same filter across the entire lightcurve
                            ("fwhm_world", float),			    # Seeing, or FWHM of the star in degrees
                            ("org_cal_mag", float),			    # This is the original calibrated magnitude before colour correction
                            ("org_cal_err", float),			    # This is the original calibrated magnitude error before colour correction
                            ("observation_id", int),		    # ID number of the observation in the database
                            ("user_id", int),			        # ID number of the user that uploaded the image to the database
                            ("target", int),			        # ID number of the target field the source has been uploaded as
                            ("flags", int),				        # Flags from the SourceExtractor added during the photometry (additionally +512 if source too bright for calibration, +1024 if too faint for calibration))
                            ("magnitude", float),			    # Uncalibrated magnitude of the star in the database
                            ("device_id", int),			        # ID number of the device the data was taken with
                            ("fits_id", int)])                  # ID number for the fits image in the database

    donotrun = 0
    try:
        with CONNECTION.cursor() as cursor:
            sql = "SELECT target_id \
            FROM photometry INNER JOIN observations \
            ON photometry.observation_id = observations.id \
            WHERE (alpha_j2000 BETWEEN %s-(%s/3600 / COS(%s * PI() / 180)) \
            AND %s+(%s/3600 / COS(%s * PI() / 180)) \
            AND delta_j2000 BETWEEN %s-%s/3600 AND %s+%s/3600) \
            ORDER BY detected_stars DESC;"

            cursor.execute(sql, (ra, 3, dec, ra, 3, dec, dec, 3, dec, 3))
            target = cursor.fetchone()
        print("Target id: {}".format(target["target_id"]))
        ## ----- if the above doesn't work, the rest of the code should not be executed ----- ##

        if not os.path.isfile(os.getcwd() + "/corrections/catalogues/{}_calib_cat.csv".format(target["target_id"])):
            # --- create file --- #
            catalogue_create(target["target_id"])
    except:
        print("Failed to create catalogue!!")
        donotrun = 1
    try:
        if (len(lightcurve) > 0) & (donotrun == 0):
            star_lc = pd.DataFrame(lightcurve)

            star_lc = star_lc[(star_lc["calibrated_magnitude"] != 0.) &
                              (star_lc["calibrated_magnitude"] != -99.) &
                              (star_lc["calibrated_error"] != 0.) &
                              (star_lc["calibrated_error"] != -99.) &
                              (star_lc["flags"] <= 4)]

            print("Reading calibration catalogue...")
            cat = pd.read_csv(os.getcwd() + "/corrections/catalogues/{}_calib_cat.csv".format(star_lc["target"].iloc[0]), index_col=0)
            print("Found {} calibration stars for the field {}.\n".format(len(cat), star_lc["target"].iloc[0]))

            cat_coords = SkyCoord(cat["alpha_j2000"], cat["delta_j2000"], frame="icrs", unit=u.degree)
            star_coord = SkyCoord(numpy.mean(lightcurve["alpha_j2000"]), numpy.mean(lightcurve["delta_j2000"]),
                                  frame="icrs", unit=u.degree)

            idx2, d2d2, d3d2 = match_coordinates_sky(star_coord, cat_coords)
            if d2d2.arcsec <= 3:
                cat = cat.drop(cat.index[idx2])
            cat = cat.reset_index(drop=True)
            cat_coords = SkyCoord(cat["alpha_j2000"], cat["delta_j2000"], frame="icrs", unit=u.degree)

            # star_lc_1 = star_lc[star_lc["user_id"] == 16]
            # star_lc_1 = star_lc_1[star_lc_1["device_id"] == 20]  # only applies to user 16.
            # star_lc_1 = star_lc_1[star_lc_1["filter"] == "I"]

            # for count, fits_id in enumerate(star_lc_1["fits_id"]):  # uncomment if using selection criteria above.
            for count, fits_id in enumerate(star_lc["fits_id"]):  # comment if using selection criteria above.
                print("Correcting image with fits_id: {}...".format(fits_id))
                star_lc = correction(fits_id, star_coord, cat_coords, cat, star_lc)
                print('done ',count+1,' of ',len(star_lc),' images.....')
                print("")

            print("\nReverting to datatype.")
            lightcurve = numpy.asarray(list(star_lc.itertuples(index=False, name=None)), dtype=datatype)

    except:
        print("Failed to correct colours!!")
        donotrun = 1
    check = numpy.isnan(lightcurve['calibrated_magnitude'])
    lightcurve = lightcurve[~check]
    return lightcurve



#old functions - now obsolete
	
#def check_filter_single(lightcurve_data, filt_list=''): 
#	if filt_list=='': filt_list=['B','V','R','I','HA']
#	filt_list=numpy.array(filt_list)
#	counter = numpy.zeros(len(filt_list))
#	for stars in lightcurve_data:
#		check = numpy.where(filt_list == stars['filter'])
#		if (len(check[0]) > 0):
#			counter[check[0][0]] = counter[check[0][0]]+1
#	print('There are ',counter,' files in the filters', filt_list)
#	return(counter)

#takes a lightcurve and filter set and returns the number of images in each filter
#def check_filter(lightcurve_data, filt_list=''): 
#	if filt_list=='': filt_list=['B','V','R','I','HA']
#	filt_list=numpy.array(filt_list)
#	counter = numpy.zeros(len(filt_list))
#        for cluster in lightcurve_data['seperated']:
#		for stars in lightcurve_data['seperated'][cluster]:
#                        check = numpy.where(filt_list == stars['filter'])
#                        if (len(check[0]) > 0):
#	                       	counter[check[0][0]] = counter[check[0][0]]+1
#	print('There are ',counter,' files in the filters', filt_list)
#        return(counter)

#def make_plot_single2(lightcurve_data,name,err='',filt_list='',symbols='',colours=''):
#	#make some default symbol and colour definitions based on the filter
#        #order of filters is Blue, Visual, Red, I-Band, H-alpha
#	if err=='': err=0
#	if filt_list=='': filt_list=['B','V','R','I','HA']
#	if symbols=='': symbols=['s','v','D','o','h']
#	if colours=='': colours=['Blue','Green','Red','Black','Magenta']
#	filt_list=numpy.array(filt_list)
#        print('starting plot...')
#	rectangle1 = [0.1,0.1, 0.8,0.8]
#	ax1 = plt.axes(rectangle1)
#	minimum=30.0
#        maximum=0.0
#	for stars in lightcurve_data:
#		check = numpy.where(filt_list == stars['filter'])
#                i = check[0][0]
#                if ((stars['calibrated_magnitude'] >= 0) and (stars['calibrated_magnitude'] <= 30) and (float(stars['flags']) <= 4.0) and (float(stars['calibrated_error']) > 0.0) and (float(stars['calibrated_error']) < 0.3)):
#			if (stars['calibrated_magnitude'] > maximum):
#        	         	maximum = float(stars['calibrated_magnitude'])
#			if (stars['calibrated_magnitude'] < minimum):
#	                 	minimum = float(stars['calibrated_magnitude'])
#                        plt.scatter(stars['date'], stars['calibrated_magnitude'], s=10.,c=colours[i], marker=symbols[i], edgecolor='black', alpha=1.0, lw=0.2)
#                        if (err == 1):
#                        	plt.errorbar(float(stars['date']), float(stars['calibrated_magnitude']), yerr=float(stars['calibrated_error']), c=colours[i], marker='', alpha=1.0, lw=0.2)
#
#	ax1.set_ylabel('Magnitudes')
#	ax1.set_xlabel('Julian Date')
#	ax1.set_ylim((maximum+0.3,minimum-0.3))
#	#plt.show()
#	plt.savefig('lightcurves/lightcurve_'+name+'.pdf', format='pdf', bbox_inches='tight', dpi=600)
#	plt.savefig('lightcurves/lightcurve_'+name+'.png', format='png', bbox_inches='tight', dpi=600)
#        plt.clf()

