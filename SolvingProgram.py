# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:13:14 2023

@author: ryanw
"""

### --- UNCUBEMAPPING DATA --- ###
# seed = input("Please enter the seed of the Universe you want to test:")
seed = 4
datapath = f'universe_{seed}' # all of the data is within a folder in this .ipynb file's directory

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# fig, axes = plt.subplots(3, 4, figsize=(12, 9)) # generate a figure to fit 3 high by 4 wide square images
# fig.subplots_adjust(wspace=0, hspace=0) # we want these squares to be adjacent to each other, with no gap
# # now to iterate over each subplot and remove the axis bars and ticks/labels
# for row in axes:
#     for ax in row:
#         for side in ['top','right','bottom','left']:
#             ax.spines[side].set_visible(False)
#         ax.tick_params(axis='both', which='both', labelbottom=False, bottom=False, left=False, labelleft=False)

# # now we load in the images and put them at their correct location
# for i, direct in enumerate(["Back", "Left", "Front", "Right", "Top", "Bottom"]): # one loop for each direction
#     img = mpimg.imread(datapath + f'/{direct}/{direct}.png') # this loads in the image from the corresponding folder
#     img_cropped = img[:-700, 900:] # crop the image to remove the axis labels/ticks
#     if i == 4: # if the Top image
#         imgplot = axes[0][1].imshow(img_cropped) # image needs to go at the top
#     elif i == 5: # if the Bottom image
#         imgplot = axes[2][1].imshow(img_cropped) # image needs to go at the bottom
#     else:
#         imgplot = axes[1][i].imshow(img_cropped) # put the image in the middle row at the correct column
        
        
import numpy as np
def cube_to_equirect(direction, u, v):
    # convert range -45 to 45 to -1 to 1
    uc = u / 45
    vc = v / 45
    if direction == "Front": # POSITIVE X
        x = 1
        y = vc
        z = -uc 
    elif direction == "Back":  # NEGATIVE X
        x = -1
        y = vc
        z = uc
    elif direction == "Top": # POSITIVE Y
        x = uc
        y = 1
        z = -vc
    elif direction == "Bottom": # NEGATIVE Y
        x = uc
        y = -1
        z = vc
    elif direction == "Left": # POSITIVE Z
        x = uc
        y = vc
        z = 1
    else: # direction == "Right": # NEGATIVE Z
        x = -uc
        y = vc
        z = -1 
    # now to convert the XYZ to spherical coordinates
    # this is using the physics convention of spherical coords!
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(z, x)
    theta = np.arccos(y / r)

    theta = theta * 180 / np.pi
    azimuth = (- azimuth + np.pi) * 360 / (2 * np.pi)
    
    return azimuth, theta

import pandas as pd


for i, direct in enumerate(["Front", "Back", "Left", "Right", "Top", "Bottom"]):
        # read the data from the .txt file into a dataframe
        stardata = pd.read_csv(datapath + f'/{direct}/Star_Data.csv', delimiter=',')  
        u = stardata["X"].to_numpy(); v = stardata["Y"].to_numpy() # convert X and Y data to "U" and "V" data
        azimuth, theta = cube_to_equirect(direct, u, v) # perform the coordinate transform
        azimuth = np.around(azimuth, decimals=4); theta = np.around(theta, decimals=4) # round to appropriate decimals
        
        df = pd.DataFrame({"Equat": azimuth, "Polar": theta}) # make a temporary DataFrame object with new coordinates
        # now overwrite the old coordinates with the new ones
        stardata['X'] = df['Equat']
        stardata["Y"] = df["Polar"]
        stardata = stardata.rename(columns={"X": "Equat", "Y": "Polar"}) # and finally change the name of the columns 
        if i == 0:
            # if this is the first iteration, write to a new DataFrame that will store all of the star data
            all_stardata = stardata
        else:
            all_stardata = pd.concat([all_stardata, stardata]) # add this face stardata to the rest of the data

all_stardata.to_csv(datapath + "/Converted_Star_Data.csv", index=False, sep=',')
# dont want to save the 'indices' of the data, and I want a space character to separate the data


for i, direct in enumerate(["Front", "Back", "Left", "Right", "Top", "Bottom"]):
        # read the data from the .txt file into a dataframe
        galaxdata = pd.read_csv(datapath + f'/{direct}/Distant_Galaxy_Data.csv', delimiter=',')  
        u = galaxdata["X"].to_numpy(); v = galaxdata["Y"].to_numpy() # convert X and Y data to "U" and "V" data
        azimuth, theta = cube_to_equirect(direct, u, v) # perform the coordinate transform
        azimuth = np.around(azimuth, decimals=4); theta = np.around(theta, decimals=4) # round to appropriate decimals
        
        df = pd.DataFrame({"Equat": azimuth, "Polar": theta}) # make a temporary DataFrame object with new coordinates
        # now overwrite the old coordinates with the new ones
        galaxdata['X'] = df['Equat']
        galaxdata["Y"] = df["Polar"]
        galaxdata = galaxdata.rename(columns={"X": "Equat", "Y": "Polar"}) # and finally change the name of the columns 
        if i == 0:
            # if this is the first iteration, write to a new DataFrame that will store all of the star data
            all_galaxdata = galaxdata
        else:
            all_galaxdata = pd.concat([all_galaxdata, galaxdata]) # add this face stardata to the rest of the data

# # now let's plot the data to see if it's worked!
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.scatter(all_galaxdata["Equat"].to_numpy(), all_galaxdata["Polar"].to_numpy(), s=0.5, c='k', lw=0);
# ax.set_xlim(0, 360); ax.set_ylim(0, 180);
# ax.invert_yaxis();


all_galaxdata.to_csv(datapath + "/Converted_Distant_Galaxy_Data.csv", index=False, sep=',')
# dont want to save the 'indices' of the data, and I want a space character to separate the data


### --- LOCAL GALAXY AND VARIABLE PERIODS --- ###
stardata = pd.read_csv(datapath + '/Converted_Star_Data.csv', delimiter=',')    # read the data from the .txt file into a dataframe

equats = stardata['Equat']    # get the equatorial positions of all of the stars
polars = stardata['Polar']     # get the polar positions of all of the stars
parallax = stardata['Parallax']    # get the parallax of the stars

indexes = [i for i, x in enumerate(parallax) if x > 0.001]   # this will return the indexes of all stars that have some parallax
# now to populate new lists with all of the equatorial/polar angles of stars that have some parallax
localequats = [equats[i] for i in indexes]  
localpolars = [polars[i] for i in indexes]

variables = stardata["Variable?"]
variableindexes = [i for i, x in enumerate(parallax) if x > 0.005 and variables[i] == 1] # we only want stars that are pretty close!
starnames = stardata['Name']; starnames = [i for i in starnames]
variablenames = [starnames[i] for i in variableindexes]
variableparallaxes = np.array([stardata["Parallax"][i] for i in variableindexes])

Vfluxes = stardata['GreenF']; variableVfluxes = np.array([Vfluxes[i] for i in variableindexes])
variableVlumins = variableVfluxes * 4 * np.pi * ((1 / variableparallaxes) * 3.086 * 10**16)**2

from astropy.timeseries import LombScargle
from glob import glob

variablepath = datapath + "/Variable_Star_Data/"
fnames = glob(variablepath + "*.csv")
freqs = np.linspace(1/120, 0.45, 10000) # frequency grid shouldn't go higher than Nyquist limit


periods = []   # initialise a list to hold all of our period data

for lightcurve in fnames:
    if lightcurve[len(variablepath):-4] in variablenames: # this checks whether the star of this lightcurve is in our variable stars
        data = pd.read_csv(lightcurve, delimiter=',') # load in the data
        time, flux = data['Time'], data['NormalisedFlux'] # just extract the columns as variables
        LS = LombScargle(time, flux) # initialize a Lomb-Scargle fitting
        power = LS.power(freqs) # calculate LS power 
        bestfreq = freqs[np.argmax(power)] # which frequency has the highest Lomb-Scargle power?
        pred = LS.model(time, bestfreq) # make a sine wave prediction at the best frequency
        periods.append(1 / bestfreq) # add each period to the list
    
periods = np.array(periods) # turn it from a list to an array


def monte_carlo(xdata, xuncs, ydata, yuncs, iterations):
    ''' Simulate many samples from data distributions and obtain a *linear* trendline with parameters and uncertainties.
    Parameters
    ----------
    x/ydata : np.array
    x/yuncs : float or np.array
        Standard deviations of our data. Can be length 1 if the uncertainty is the same across data, or an array for unique
        SDs
    '''
    # initialise arrays to store our data in
    grads = np.zeros(iterations)
    yints = np.zeros(iterations)
    x_rand = np.zeros(len(xdata))
    y_rand = np.zeros(len(xdata))
    
    # if our uncertainty is a scalar, make it an array with N times that value (N being the length of our data array)
    if np.size(xuncs) == 1:
        xuncs = np.ones(len(xdata)) * xuncs
    if np.size(yuncs) == 1:
        yuncs = np.ones(len(ydata)) * yuncs
    
    # now to perform n=iterations random samples of our data distributions
    for i in range(iterations):
        for j in range(len(xdata)):
            # generate a random normal variable for each of our XY data points
            x_rand[j] = np.random.normal(xdata[j], xuncs[j])
            y_rand[j] = np.random.normal(ydata[j], yuncs[j])
        # now fit a line to our random data. A 1-dimensional polynomial is just a straight line!
        grads[i], yints[i] = np.polyfit(x_rand, y_rand, 1)
    
    # now get the statistics of our *iterations* number of trendline parameters
    meangrad = np.mean(grads[:i])
    SDgrad = np.std(grads[:i])
    meanyint = np.mean(yints[:i])
    SDyint = np.std(yints[:i])
    return np.array([meangrad, SDgrad, meanyint, SDyint])

lowperiodindexes = [i for i, x in enumerate(periods) if 15 <= x <= 30]
# we need to do another step to remove the outlier(s) at the bottom. Check to see if lumins for each index are high
# enough luminosity:
lowperiodindexes = [lowperiodindexes[i] for i, x in enumerate(variableVlumins[lowperiodindexes]) if x > 10**25]
lowperiods = periods[lowperiodindexes]
lowperiodlumins = np.log10(variableVlumins[lowperiodindexes]) # get lumins in log scale
lowperiodunc = 0.01 / np.log(10)

# now, lets fit a linear trend to this data. We can do this with a polynomial fit of one degree from numpy:
shortgradient, shortgradSD, shortyint, shortyintSD = monte_carlo(lowperiods, 0.1, lowperiodlumins, lowperiodunc, 20000)
print(f"Short grad = {round(shortgradient, 3)}±{round(shortgradSD, 3)}; Short y-int = {round(shortyint, 2)}±{round(shortyintSD, 2)}")


longerperiodindexes = [i for i, x in enumerate(periods) if 30 <= x <= 52]
# we need to do another step to remove the outlier(s) at the bottom. Check to see if lumins for each index are high
# enough luminosity:
longerperiodindexes = [longerperiodindexes[i] for i, x in enumerate(variableVlumins[longerperiodindexes]) if x > 10**23]
longerperiods = periods[longerperiodindexes]
longerperiodlumins = np.log10(variableVlumins[longerperiodindexes]) # get lumins in log scale
longerperiodunc = 0.01 / np.log(10)

# now, lets fit a linear trend to this data. We can do this with a polynomial fit of one degree from numpy:
longergradient, longergradSD, longeryint, longeryintSD = monte_carlo(longerperiods, 0.1, longerperiodlumins, longerperiodunc, 20000)
print(f"Longer grad = {round(longergradient, 3)}±{round(longergradSD, 3)}; Longer y-int = {round(longeryint, 2)}±{round(longeryintSD, 2)}")



### --- IDENTIFYING GALAXIES --- ###

# first, lets define the points for clustering in a suitable format. the clustering algorithm needs 'n' points, 
# where each point has a coordinate [x, y]. So our final array should look like: array = [[x1,y1], [x2,y2], ...]
coords = np.ndarray((len(equats), 2))  # set up an empty array of the correct length and dimension (Nx2)
for i, equat in enumerate(equats):
    coords[i] = [equat, polars[i]] # populate each element of the array with [x, y]
    
import hdbscan

# much of this code block is identical to the one before, so I'm going to be slack with the commenting
clustering = hdbscan.HDBSCAN(min_samples=50, min_cluster_size=500).fit(coords) # want the smallest galaxy to have 500 stars
labels = clustering.labels_

import os
directory = os.path.abspath("") + f"/{datapath}"
newdir = directory + "/Star_Clusters"
if not os.path.exists(newdir):
    os.makedirs(newdir)

for clust in range(0, max(labels) + 1):
    indices = np.where(labels == clust)  # gets the indices of this galaxy's stars with respect to the stardata struct
    data = stardata.iloc[indices] # find the stars corresponding to the found indices
    
    Xk = coords[labels == clust] # get the positions of all of the stars in the galaxy
    xcenter = np.mean(Xk[:, 0]); ycenter = np.mean(Xk[:, 1]) # rough center points of each galaxy
    
    # now, I want to name the clusters like "X{equat}-Y{polar}-N{population}":
    clustername = 'X'+"%05.1f"%xcenter +'-Y'+"%05.1f"%ycenter+'-N'+str(len(Xk)) # generates cluster name
    # finally, write the data to a file defined by clustername
    data.to_csv(datapath + f'/Star_Clusters/{clustername}.csv', index=None, sep=',')
    
# - Identifying Galaxy Clusters - #
positions = np.ndarray((max(labels), 3))
names = np.ndarray((max(labels)), dtype=object)

for clust in range(0, max(labels)):
    indices = np.where(labels == clust)  # gets the indices of this galaxy's stars with respect to the stardata struct
    data = stardata.iloc[indices] # find the stars corresponding to the found indices
    
    Xk = coords[labels == clust] # get the positions of all of the stars in the galaxy
    xcenter = np.mean(Xk[:, 0]); ycenter = np.mean(Xk[:, 1]) # rough center points of each galaxy
    meanvel = np.mean(data['RadialVelocity']) # get the mean velocity of the stars within the galaxy
    positions[clust] = [xcenter, ycenter, meanvel]
    # now, I want to name the clusters like "X{equat}-Y{polar}-N{population}":
    clustername = 'X'+"%05.1f"%xcenter +'-Y'+"%05.1f"%ycenter+'-N'+str(len(Xk)) # generates cluster name
    names[clust] = clustername
    
# much of this code block is identical to the one before, so I'm going to be slack with the commenting
closeclusters = hdbscan.HDBSCAN().fit(positions) # it seems to cluster optimally with no set parameters
labels = closeclusters.labels_

newdir = directory + "/Close_Galaxy_Clusters"
if not os.path.exists(newdir):
    os.makedirs(newdir)
    
for clust in range(0, max(labels) + 1):
    Xk = positions[labels == clust] # get the positions of all of the galaxies in the cluster
    xcenter = np.mean(Xk[:, 0]); ycenter = np.mean(Xk[:, 1]) # rough center points of each galaxy
    
    # now, I want to name the clusters like "GC-X{equat}-Y{polar}-N{population}":
    clustername = 'GC-X'+"%05.1f"%xcenter +'-Y'+"%05.1f"%ycenter+'-N'+str(len(Xk)) # generates cluster name
    galaxnames = names[labels == clust] # get the names of the galaxies within this cluster
    # finally, write the data to a file defined by clustername
    with open(datapath + f'/Close_Galaxy_Clusters/{clustername}.txt', 'w') as file: # open/create this file...
        for name in galaxnames: # for each galaxy in the cluster...
            file.write(str(name)+'\n') # ...write the galaxy name, and then end the line

# - Finding Distant Galaxy Clusters - #
distantdata = pd.read_csv(datapath + '/Converted_Distant_Galaxy_Data.csv', delimiter=',')

distantequats = distantdata['Equat']    # get the equatorial positions of all of the distant galaxies
distantpolars = distantdata['Polar']     # get the polar positions of all of the distant galaxies
distantvels = distantdata['RadialVelocity']  # import the velocity data

distantcoords = np.ndarray((len(distantequats), 3))  # set up an empty array of the correct length and dimension (Nx3)
for i, equat in enumerate(distantequats):
    distantcoords[i] = [equat, distantpolars[i], distantvels[i]] # populate each element of the array with [x, y, v]

# much of this code block is identical to the one before, so I'm going to be slack with the commenting
distantclusters = hdbscan.HDBSCAN().fit(distantcoords)
labels = distantclusters.labels_

newdir = directory + "/Distant_Galaxy_Clusters"
if not os.path.exists(newdir):
    os.makedirs(newdir)
clusterCoords = np.ndarray((max(labels) + 1, 2))

for clust in range(0, max(labels) + 1):
    indices = np.where(labels == clust)  # gets the indices of this cluster's galaxies with respect to distantdata
    data = distantdata.iloc[indices] # find the galaxies corresponding to the found indices
    
    Xk = distantcoords[labels == clust] # get the positions of all of the stars in the cluster
    xcenter = np.mean(Xk[:, 0]); ycenter = np.mean(Xk[:, 1]) # rough center points of each cluster
    
    clusterCoords[clust] = (xcenter, ycenter)
    
    # now, I want to name the clusters like "DC-X{equat}-Y{polar}-N{population}":
    clustername = 'DC-X'+"%05.1f"%xcenter +'-Y'+"%05.1f"%ycenter+'-N'+str(len(Xk)) # generates cluster name
    # finally, write the data to a file defined by clustername
    data.to_csv(datapath + f'/Distant_Galaxy_Clusters/{clustername}.csv', index=None, sep=',')
    
    
### --- DISTANCE LADDER I --- ###
GalaxyNames = []
for clusterFile in os.listdir(datapath + '/Star_Clusters'):
    GalaxyNames.append(clusterFile[:-4]) # this gets the name of the cluster, without the `.txt' file extension
    
    
freqs = np.linspace(1/120, 0.45, 1000)
distance_data = pd.DataFrame({'Name': GalaxyNames})

PL_dists = np.zeros(len(GalaxyNames))
PL_unc = np.zeros(len(GalaxyNames))
for num, name in enumerate(GalaxyNames): # iterate over the identified galaxies
    galaxdata = pd.read_csv(datapath + f'/Star_Clusters/{name}.csv', delimiter=',')
    # now to isolate the variable stars
    variables = galaxdata['Variable?']
    variableindexes = [i for i, x in enumerate(variables) if x == 1]
    variablenames = galaxdata['Name'][variableindexes].to_numpy()
    periods = np.zeros(len(variablenames)) # initialise
    # now calculate the period of each variable star via LS periodogram
    for i, star in enumerate(variablenames):
        photometryData = pd.read_csv(datapath + f"/Variable_Star_Data/{star}.csv", delimiter=',')
        
        time, flux = photometryData['Time'], photometryData['NormalisedFlux'] # just extract the columns as variables
        LS = LombScargle(time, flux) # initialize a Lomb-Scargle fitting
        power = LS.power(freqs) # calculate LS power 
        bestfreq = freqs[np.argmax(power)] # which frequency has the highest Lomb-Scargle power?
        pred = LS.model(time, bestfreq) # make a sine wave prediction at the best frequency
        periods[i] = 1 / bestfreq # add each period to the list
        
    intr_lumin = np.zeros(len(periods)) # initialise
    lumin_err = np.zeros(len(periods))
    # time to work out intrinsic luminosities via the PL relations
    for i, period in enumerate(periods): 
        if 20 <= period <= 35:
            intr_lumin[i] = 10**(-0.086 * period + 28.25)
            AB = 0.087 * period * np.sqrt((0.002 / 0.087)**2 + (0.1 / period)**2)
            AB_C = np.sqrt(AB**2 + 0.04**2)
            lumin_err[i] = intr_lumin[i] * (np.log(10) * AB_C)
        if 35 <= period <= 55:
            intr_lumin[i] = 10**(-0.087 * period + 33.72)
            AB = 0.087 * period * np.sqrt((0.002 / 0.087)**2 + (0.1 / period)**2)
            AB_C = np.sqrt(AB**2 + 0.08**2)
            lumin_err[i] = intr_lumin[i] * (np.log(10) * AB_C)
        elif 80 <= period <= 110:
            intr_lumin[i] = 10**(-0.087 * period + 34.35)
            AB = 0.087 * period * np.sqrt((0.001 / 0.087)**2 + (0.1 / period)**2)
            AB_C = np.sqrt(AB**2 + 0.06**2)
            lumin_err[i] = intr_lumin[i] * (np.log(10) * AB_C)
            
    # now to finally compare intr_lumin against the fluxes to obtain distances
    greenFluxes = galaxdata['GreenF'][variableindexes].to_numpy()
    distances = []
    distance_unc = []
    
    if len(intr_lumin) > 10: # we only want to look at galaxies with reasonably high sample size of variables
        for i, lumin in enumerate(intr_lumin):
            if lumin > 0: # this is to avoid noisy data
                distance_m = np.sqrt(lumin / (4 * np.pi * greenFluxes[i]))
                distance_pc = distance_m / (3.086 * 10**16)
                distances.append(distance_pc)
                
                lumin_unc = distance_m**2 * np.sqrt((lumin_err[i] / lumin)**2 + (0.01)**2)
                distance_pc_unc = abs(distance_m * 0.5 * lumin_unc / distance_m**2) / (3.086*10**16)
                distance_unc.append(distance_pc_unc)
        # now we need to remove any (inevitable) outliers from the data
        q1, q3 = np.percentile(distances, [25, 75], interpolation='midpoint')
        IQR = q3 - q1
        # the below checks if the data point is within 2.7 standard deviations of the mean
        upper = q3 + 1.5*IQR
        lower = q1 - 1.5*IQR
        # now we only care about data within 2.7 STDs 
        distances = [dist for dist in distances if lower <= dist <= upper]
        distance_unc = [distance_unc[i] for i, x in enumerate(distances) if lower <= x <= upper]
    PL_dists[num] = np.mean(distances)
    PL_unc[num] = np.mean(distance_unc)
    
distance_data['PL_distance'] = PL_dists
distance_data['PL_unc'] = PL_unc

# - Main Sequence Fitting - #
allstardata = pd.read_csv(datapath + '/Converted_Star_Data.csv', delimiter=',')    # read the data from the .txt file into a dataframe

parallax = allstardata['Parallax']    # get the parallax of the stars

localindex = [i for i, x in enumerate(parallax) if x > 0.007]
localVflux = np.array(allstardata["GreenF"])[localindex]
localBflux = np.array(allstardata["BlueF"])[localindex]
BV = np.log10(localVflux / localBflux) # B-V colour index 

localVlumin = localVflux * 4 * np.pi * (1 / np.array(allstardata["Parallax"][localindex]) * 3.086 * 10**16)**2

MS_dist = np.zeros(len(GalaxyNames))
MS_unc = np.zeros(len(GalaxyNames))
for num, name in enumerate(GalaxyNames):
    galaxdata = pd.read_csv(datapath + f'/Star_Clusters/{name}.csv', delimiter=',')
    galaxVflux = np.array(galaxdata["GreenF"])
    galaxBflux = np.array(galaxdata["BlueF"])
    galaxBV = np.log10(galaxVflux / galaxBflux) # B-V colour index 
    
    offset = max(np.log10(localVlumin)) - max(np.log10(galaxVflux))
    
    dist_m = np.sqrt(10**offset / (4 * np.pi))
    dist_pc = dist_m / (3.086 * 10**16)
    
    dist_pc_unc = (dist_m**2 * np.log(10) * np.sqrt(2) * 0.01 / (4 * np.pi)) * 0.5 * dist_m / (dist_m**2 * (3.086*10**16))
    
    MS_dist[num] = dist_pc
    MS_unc[num] = dist_pc_unc

distance_data['MS_distance'] = MS_dist
distance_data['MS_unc'] = MS_unc
distance_data.to_csv(datapath + '/Galaxy_Distances.csv', index=None, sep=',')


### --- DISTANCE LADDER II --- ###
xrayData = pd.read_csv(datapath + '/Flash_Data.csv', delimiter=',')
galaxDists = pd.read_csv(datapath + '/Galaxy_Distances.csv', delimiter=',')

u = xrayData["X"].to_numpy(); v = xrayData["Y"].to_numpy() # convert X and Y data to "U" and "V" data
direct = xrayData["Direction"].to_numpy()

azimuth = np.zeros(len(u)); theta = np.zeros(len(u))
for i in range(len(u)):
    temp_azimuth, temp_theta = cube_to_equirect(direct[i], u[i], v[i]) # perform the coordinate transform
    azimuth[i] = np.around(temp_azimuth, decimals=4); theta[i] = np.around(temp_theta, decimals=4) # round to appropriate decimals

xrayData['Equat'] = azimuth
xrayData["Polar"] = theta

equats, polars, dists = np.zeros(len(galaxDists)), np.zeros(len(galaxDists)), np.zeros(len(galaxDists))

for i in range(len(galaxDists)):
    name = galaxDists['Name'][i]
    equats[i] = name[1:6]
    polars[i] = name[8:13]
    dists[i] = galaxDists['MS_distance'][i]
    
closeflashes = xrayData.loc[xrayData['Photon-Count'] > 1e6]

lumins = []
delta = 5
for i, flash in closeflashes.iterrows():
    for j, equat in enumerate(equats):
        if (flash['Equat'] - delta <= equat <= flash['Equat'] + delta) and (flash['Polar'] - delta <= polars[j] <= flash['Polar'] + delta):
            lumin = flash['Photon-Count'] * 4 * np.pi * dists[j]**2
            lumins.append(lumin)
lumins = np.array(lumins)

# - Hubble Constant and Distant Galaxies - #
base_lumin = np.mean(lumins)
xraydists = np.zeros(len(xrayData))
for i, flash in xrayData.iterrows():
    xraydists[i] = np.sqrt(base_lumin / (4 * np.pi * flash['Photon-Count']))

distGalaxData = pd.read_csv(datapath + '/Converted_Distant_Galaxy_Data.csv', delimiter=',')
speeds = np.zeros(len(xrayData))
delta = 0.5
for i, flash in xrayData.iterrows():
    for j, equat in enumerate(distGalaxData['Equat']):
        if (flash['Equat'] - delta <= equat <= flash['Equat'] + delta) and (flash['Polar'] - delta <= distGalaxData['Polar'][j] <= flash['Polar'] + delta):
            speeds[i] = distGalaxData['RadialVelocity'][j]

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(xraydists, speeds)
print(xraydists)
print(speeds)

grad, yint = np.polyfit(xraydists, speeds, 1)
xvals = np.array([0, max(xraydists)])

print(f"m = {round(grad * 1e6, 2)} km/s/Mpc; y-int = {round(yint, 2)} km/s")

distGalaxData['Distance'] = (distGalaxData['RadialVelocity'].to_numpy() - yint) / grad

with open(datapath + '/answer.txt', "w") as file:
    file.write(f"H_0 = {round(grad * 1e6, 2)} km/s/Mpc; y-int = {round(yint, 2)} km/s")