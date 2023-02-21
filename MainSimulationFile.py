# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:26 2022

@author: ryanw

"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import pandas as pd
from time import time
from tqdm import tqdm     # this is a progress bar for a for loop
import MiscTools as misc
from BlackHole import BlackHole
from Galaxy import Galaxy
from GalaxyCluster import GalaxyCluster
from Star import Star
from Nebula import Nebula
from Universe import Universe
from HTMLSite import Site

from argparse import ArgumentParser

def plot_all_dopplers(galaxies):
    ''' Plot the radial velocities of a list of Galaxy objects onto an image. Mainly to be used for troubleshooting.
    '''
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1]})
    for galaxy in galaxies:
        galaxy.plot_doppler(fig, ax, cbar_ax, blackhole=True)
def plot_all_2d(galaxies, spikes=False, radio=False):
    ''' Plot the positions, colours and brightness of a list of Galaxy objects onto an image. Mainly to be used for troubleshooting.
    '''
    fig, ax = plt.subplots()
    for galaxy in galaxies:
        galaxy.plot_2d(fig, ax, spikes=spikes, radio=radio)
def plot_all_3d(galaxies):
    ''' Plot 3D galaxies from a list of Galaxy objects. Mainly to be used for troubleshooting.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for galaxy in galaxies:
        galaxy.plot_3d(ax, camera=False)

class UniverseSim(object):
    ''' The class that handles generating the universe and saving subsequent data. Usually, this is the only class
    that should be interacted with by the user. 
    '''
    def __init__(self, numclusters, hubble=None, seed=None, blackholes=True, darkmatter=True, mode="Normal",
                 homogeneous=True, isotropic=True, rotvels="Boosted", event="flash"):
        ''' Generates a Universe object and imports important data from it. 
        Parameters
        ----------
        numclusters : int
            The number of GalaxyCluster objects to populate the universe with.
        hubble : float
            The value of the Hubble constant in units of km/s/Mpc.
        seed : int
            A random seed for which to apply to random number generation in the program.
        mode : str
            One of {"Comprehensive", "Normal", "Basic"} which chooses how complicated the data analysis will be. 
            Comprehensive - some galaxies will and won't have darkmatter/blackholes, and have many stars.
            Normal - all galaxies either will or won't have darkmatter/blackholes, and have many stars.
            Basic - all galaxies either will or won't have darkmatter/blackholes, and have few stars.
        homogeneous : bool
            Whether the universe is homogeneous (approx constant density with distance) or not (exp. increasing, then decreasing)
        isotropic : bool
            Whether the local galaxy will 'absorb light from distant galaxies' or not. Makes it so galaxy clusters
            are less likely close to polar = 0 (the galactic plane of the local galaxy)
        rotvels : str
            One of {"Normal", "Boosted"}, which dictates whether rotation curves are normal, or have arbitrarily (and unphysically) 
            boosted velocity magnitudes.
        event : str
            One of {"flash", "supernova"} which determines the type of bright event that allows the distance ladder formulation. The
            event type determines the format of the output data (photon count of flash, or W/m^2 respectively)
        '''
        self.seed = seed if seed != None else int(np.random.uniform(0, 9999)) # randomly choose a <=4 digit seed if one isn't given
        np.random.seed(seed)
        self.universe = Universe(450000, numclusters, hubble, blackholes=blackholes, darkmatter=darkmatter, complexity=mode,
                                 homogeneous=homogeneous, isotropic=isotropic, rotvels=rotvels, event=event)
        self.hasblackhole = blackholes 
        self.hasdarkmatter = darkmatter
        self.mode = mode
        self.homogeneous = homogeneous 
        self.isotropic = isotropic
        self.eventType = event
        self.hubble = self.universe.hubble
        self.galaxies, self.distantgalaxies = self.universe.get_all_galaxies()
        self.allgalaxies = self.galaxies + self.distantgalaxies
        self.flashes = self.universe.flashes
        self.starpositions = self.universe.get_all_starpositions()
        self.numstars = len(self.starpositions[0])
        self.blackholes = self.universe.get_blackholes()  
        self.datadirectory, self.cubemapdirectory = False, False
        
        # now generate names for each star
        width = int(-(-np.log10(self.numstars) // 1))   # find the "size" of the number of stars, and round up to the closest decimal
        self.starnames = np.array([f"S{i:0{width}d}" for i in range(1, self.numstars + 1)])   # generate pretty useless names for each of the stars
    
    def create_directory(self):
        ''' Creates the folder for the universe data to be stored in. Will only be created prior to trying to save data. 
        '''
        # first, initialise the directory where all data will be saved
        self.directory = os.path.dirname(os.path.realpath(__file__))    # this is where this .py file is located on the system
        subdirectory = f"/Datasets/universe_{self.seed}/"
        self.datadirectory = self.directory + subdirectory
        if os.path.exists(self.datadirectory):  # if this directory exists, we need to append a number to the end of it
            i = 1
            while os.path.exists(self.datadirectory):   # this accounts for multiple copies that may exist
                self.datadirectory = self.datadirectory + f" ({i})"   # add the number to the end
                i += 1
            os.makedirs(self.datadirectory)     # now create the duplicate directory with the number on the end
        else:
            os.makedirs(self.datadirectory)     # if directory doesn't exist, create it

        # answers
        subdirectory = f"/Answers/universe_{self.seed}/"
        self.answerdirectory = self.directory + subdirectory
        if os.path.exists(self.answerdirectory):  # if this directory exists, we need to append a number to the end of it
            i = 1
            while os.path.exists(self.answerdirectory):   # this accounts for multiple copies that may exist
                self.answerdirectory = self.answerdirectory + f" ({i})"   # add the number to the end
                i += 1
            os.makedirs(self.answerdirectory)     # now create the duplicate directory with the number on the end
        else:
            os.makedirs(self.answerdirectory)     # if directory doesn't exist, create it

        
    def create_cubemap_directory(self, proj="Cube"):
        ''' Creates the directional folders needed for the cubemap projection data/images.
        A folder is created for each of ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right'].
        Parameters
        ----------
        proj : str
            One of {"Cube", "DivCube"}, to determine how many subfolders to make. If "DivCube" is selected, 
            folders 'A1' through to 'F6' are created to divide each directional image into 36 squares of 15x15 deg each.
        '''
        if not self.datadirectory: # if the parent data directory hasnt been created,
            self.create_directory() # it must be created
        directions = ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right']
        for direction in directions: # now make a folder for each direction
            os.makedirs(self.datadirectory + f'/{direction}')
            if proj == "DivCube":
                for X in ["A", "B", "C", "D", "E", "F"]:
                    for Y in range(1, 7):
                        os.makedirs(self.datadirectory + f'/{direction}/{X}{Y}')
        self.cubemapdirectory = True # and finally set it so that we know the cubemap directional directories have been made
    
    def plot_universe(self, spikes=True, radio=False, pretty=True, planetNeb=False, save=False, cubemap=False):
        ''' Plot all of the stars and distant galaxies in the universe onto a rectangular mapping of the inside of the 
        observable universe sphere. X-axis units are in "Equatorial Angle (degrees)", with Y-axis units in "Polar Angle (degrees)."
        Parameters
        ----------
        spikes : bool
            If true, add diffraction spikes to stars according to their apparent brightness
        radio : bool
            If true, plot the black hole radio lobes overlaid onto the universe image.
        pretty : bool
            Whether to plot galaxy nebulosity on top of galaxies. Warning: high RAM usage!
        planetNeb : bool
            Whether to generate and plot planetary nebulae in the local/close galaxies
        save : bool
            If true, returns the matplotlib figure object so that it may be saved further downstream. 
        Returns
        -------
        fig : matplotlib figure object (if "save" parameter is True)
        '''
        if not cubemap: # we want a single, all sky image
            fig, ax = plt.subplots()
        else: # we want 6, directional images of a cubemap
            figAxes = []
            for i in range(6):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.set_xlim(-45, 45); ax.set_ylim(-45, 45)    # equatorial angle goes from 0->360, polar 0->180
                ax.set_facecolor('k')   # space has a black background, duh
                ax.set_aspect(1)    # makes it so that the figure is twice as wide as it is tall - no stretching!
                # fig.tight_layout()
                ax.set_xlabel("X Position (degrees)")
                ax.set_ylabel("Y Position (degrees)")
                ax.grid(linewidth=0.1)
                
                figAxes.append([fig, ax])
        stars = self.starpositions
        x, y, z, colours, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        equat, polar, radius = misc.cartesian_to_spherical(x, y, z)
        colours = colours[:]    # this fixes an issue when rerunning this program since we want to make a copy of the colours array
        
        for i, blackhole in enumerate(self.blackholes):     # first, plot the black holes at their positions
            if blackhole == False:
                continue
            BHequat, BHpolar, distance = self.allgalaxies[i].spherical  # get the coords of each black hole
            if cubemap:
                BHx, BHy, BHz = self.allgalaxies[i].cartesian
                uc, vc, index = misc.cubemap(BHx, BHy, BHz)
            BHcolour = blackhole.get_BH_colour()
            BHscale = blackhole.get_BH_scale() / (0.05 * distance)  # determine the scale of the black hole marker from its intrinsic brightness and distance
            if spikes == True and BHscale > 2.5:    # then we want to plot diffraction spikes on the black hole
                spikesize = BHscale / 2
                if not cubemap:
                    ax.errorbar(BHequat, BHpolar, yerr=spikesize, xerr=spikesize, ecolor=BHcolour, fmt='none', elinewidth=0.3, alpha=0.5)
                else:
                    self.cubemap_plot(uc, vc, index, 0, BHcolour, figAxes, spikes=spikesize)
            if save == True:    # we want to get rid of the circle outline on the marker if we're saving the image
                if not cubemap:
                    ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale, linewidths=0)
                else:
                    self.cubemap_plot(uc, vc, index, BHscale, BHcolour, figAxes)
            else:   # matplotlib can automatically scale markers when zooming, so we don't need to worry about the circle outlines. 
                if not cubemap:
                    ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale)
                else:
                    self.cubemap_plot(uc, vc, index, BHscale, BHcolour, figAxes)
        
        # now, we want to plot all of the stars in the universe. Firstly, we need to determine how large they should appear
        j = np.zeros(len(self.galaxies) + 1)    # initialise an array to store galaxy populations
        for i, galaxy in enumerate(self.galaxies):
            pop = len(galaxy.get_stars()[0])
            j[i+1] = int(pop)       # add this galaxy's population to the aforementioned array
        cumpops = [int(sum(j[:i + 1])) if i != 0 else int(j[i]) for i in range(len(j))]     # determine the cumulative population of the galaxies in their order
        # the cumulative population is needed to keep track of which star is being looked at in the next step
        scales = [[scales[i+cumpops[k]] / (0.05 * radius[i+cumpops[k]]) for i in range(len(galaxy.get_stars()[0]))] 
                  for k, galaxy in enumerate(self.galaxies)]    # get the scale of each star, and divide it by some function of its distance to get its apparent brightness
        scales = [scale for galaxy in scales for scale in galaxy]   # the above operation produces nested lists, so this step flattens the list
        if spikes == True:  # now to plot diffraction spikes on the bright stars
            if not cubemap:
                brightequat, brightpolar, brightscale, brightcolour = [], [], [], np.empty((0, 3))
            else:
                brightX, brightY, brightZ, brightscale, brightcolour = [], [], [], [], np.empty((0, 3))
            for i, scale in enumerate(scales):
                if scale > 2.5:     # we want spikes on this star!
                    if not cubemap:
                        brightequat += [equat[i]]
                        brightpolar += [polar[i]]
                    else:
                        brightX += [x[i]]
                        brightY += [y[i]]
                        brightZ += [z[i]]
                    brightscale = brightscale + [scale / 4]
                    brightcolour = np.append(brightcolour, [colours[i]], axis=0)
            # now plot makeshift diffraction spikes with no marker (so that the marker can accurately be plotted later)
            if not cubemap:
                ax.errorbar(brightequat, brightpolar, yerr=brightscale, xerr=brightscale, ecolor=brightcolour, fmt='none', elinewidth=0.3)
            else:
                brightX, brightY, brightZ, brightscale = np.array(brightX), np.array(brightY), np.array(brightZ), np.array(brightscale)
                uc, vc, index = misc.cubemap(brightX, brightY, brightZ)
                for i in range(len(uc)): # now plot each of the bright stars individually
                    self.cubemap_plot(uc[i], vc[i], index[i], 0, brightcolour[i], figAxes, spikes=brightscale[i])
                    
        scales = [3 if scale > 3 else abs(scale) for scale in scales]   # limit the maximum size of stars so that an abnormally close star doesnt cover the image
        
        # now we obtain the distant galaxy positions and data, and append them to the star arrays (plotting all in one go saves some time)
        DGspherical = np.array([galaxy.spherical for galaxy in self.distantgalaxies])
        if not cubemap:
            DGequat, DGpolar, DGdists = DGspherical[:, 0], DGspherical[:, 1], DGspherical[:, 2]
            equat, polar = np.append(equat, DGequat), np.append(polar, DGpolar)     # append the distant galaxy data to the end of the star data
        else:
            DGdists = DGspherical[:, 2]
            DGcartesian = np.array([galaxy.cartesian for galaxy in self.distantgalaxies])
            DGx, DGy, DGz = DGcartesian[:, 0], DGcartesian[:, 1], DGcartesian[:, 2]
            x, y, z = np.append(x, DGx), np.append(y, DGy), np.append(z, DGz)
            
        DGscales = 1 / (0.0001 * DGdists)   # we want to artifically make distant galaxies a bit larger than stars, since they *should* be brighter and bigger
        scales.extend(DGscales)
        colours.extend([[1.0, 0.8286, 0.7187] for i in range(len(DGscales))])
        # for scale in DGscales:
        #     scales.append(scale)
        #     colours.append([1.0000, 0.8286, 0.7187])    # this is a nice enough colour to show distant galaxies as. Plotting them based on their actual colour would be too expensive
        
        if not cubemap:
            if save == True:    # as with earlier, plot with no circle outline if saving
                ax.scatter(equat, polar, s=scales, c=colours, linewidths=0)
            else:
                ax.scatter(equat, polar, s=scales, c=colours)
            ax.set_xlim(0, 360); ax.set_ylim(0, 180)    # equatorial angle goes from 0->360, polar 0->180
            ax.set_facecolor('k')   # space has a black background, duh
            ax.set_aspect(1)    # makes it so that the figure is twice as wide as it is tall - no stretching!
            fig.tight_layout()
            ax.invert_yaxis()   # polar angle of 0 is at the top, 180 at the bottom
            ax.set_xlabel("Equatorial Angle (degrees)")
            ax.set_ylabel("Polar Angle (degrees)")
        else:
            scales = np.array(scales)
            colours = np.array(colours)
            uc, vc, index = misc.cubemap(x, y, z)
            self.cubemap_plot(uc, vc, index, scales, colours, figAxes)
        
        if not cubemap:
            figAxes = [fig, ax]
            method = "AllSky"
        else:
            method = "Cube" 
        if pretty or planetNeb:
            # let's plot the galaxy nebulosity for the close galaxies, or planetary nebulae
            import gc
            gc.enable()
            for i, galaxy in enumerate(self.galaxies):
                if planetNeb:
                    # we're going to plot planetary nebulae for each close white dwarf that is relatively hot (which i rationalise
                    # is a young white dwarf star, and needs a nebula accordingly)
                    if galaxy.spherical[2] < 1500:
                        for j, star in enumerate(galaxy.stars):
                            if star.species == "WDwarf" and star.temperature >= 18000:
                                pos = [galaxy.starpositions[0][j], galaxy.starpositions[1][j], galaxy.starpositions[2][j]]
                                planetaryNeb = Nebula('ring', pos, cartesian=True, reduction=True)
                                planetaryNeb.plot_nebula(figAxes=figAxes, method=method)
                if pretty:
                    # here is where we generate and plot the nebulosity for each close galaxy
                    if galaxy.spherical[2] < 20000: # if the galaxy is closer than 20kpc
                        local = True if i == len(self.galaxies) - 1 else False # the local galaxy is the last in the list
                        galaxy.plot_nebulosity(figAxes, method=method, localgalaxy=local)
                        gc.collect()
                
            
        if radio == True:   # plot the radio overlay
            self.plot_radio(figAxes)
        
        if save == True:    # close the figure (so it doesnt pop up during the run) and return the figure to save later.
            if not cubemap:
                fig, ax = figAxes
                return fig, ax
            else:
                return figAxes
    
    def cubemap_plot(self, uc, vc, index, scales, colours, figAxes, spikes=None):
        ''' Plots the points on each cube face for the cube mapping. 
        Parameters
        ----------
        uc, vc : np.array
            horiz, vert coords (respectively) of each point
        index : np.array
            indices of each point corresponding to their projected cube face
        scales, colours : np.array
            size and colour of each point on the image
        figAxes: list of matplotlib objects
            The 6 figure/ax matplotlib objects corresponding to the plot of each cube face. Must be arranged like:
                [[fig1, ax1], [fig2, ax2], ..., [fig6, ax6]]
        spikes : float
            None by default if no diffraction spikes, but input a number to get diffraction spikes on these coords
            Spiked coords MUST be plotted one at a time, otherwise code changes are needed
        '''
        for i in range(6):
            x, y = uc[index == i], vc[index == i] # get all coords of points on this cube face
            if spikes == None: # don't need to worry about diffraction spikes, just scatter the points
                if uc.size == 1:
                    figAxes[i][1].scatter(x, y, s=scales, color=colours, linewidths=0)
                else:
                    figAxes[i][1].scatter(x, y, s=scales[index == i], color=colours[index == i], linewidths=0)
            else: # we have spikes, so need to use error bar
                figAxes[i][1].errorbar(x, y, yerr=spikes, xerr=spikes, ecolor=colours, fmt='none', elinewidth=0.3)
    
    def plot_radio(self, figAxes, method="AllSky"):
        ''' Plot the radio contours of the SMBH emission onto a 2D sky plot. 
        Parameters
        ----------
        figAxes : list (or None)
            List in the form of [fig, ax] (if AllSky projection), or [[fig1, ax1], [fig2, ax2],...,[fig6, ax6]] if cubemapped.
            If you want a new generation, input just None
        method : str
        '''
        for galaxy in self.galaxies:
            if galaxy.blackhole == False or galaxy.blackhole.BHradio == False or galaxy == self.galaxies[-1]:
                # we can't plot radio contours if there's no black hole, or if there's no radio lobes from a galaxy's BH
                # I also don't want to plot any radio contours for the local galaxy
                continue
            galaxy.plot_radio_contour(figAxes, method=method, plot=True, scatter=False, data=False, thin=True)
        
    def plot_doppler(self, log=True, save=False):
        ''' Plot the radial velocities of all of the stars and distant galaxies in the universe in terms of a colour scale,
        where the objects are at their positions in the sky. Similar to the "plot_universe()" function. 
        Parameters
        ----------
        log : bool
            If true (by default), plots the radial velocities with a logarithmically scaled colourbar. If False (I don't recommend),
            plots with a linearly scaled colourbar.
        save : bool
            If true, returns the figure so that it may be saved later. 
        Returns
        -------
        figure : matplotlib figure object
            If save==True, closes the figure and returns it so that it may be saved later on. 
        '''
        # generate a figure and colourbar with width ratio of 30:1
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1], 'wspace': 0.1, 
                                                             'height_ratios': [1]})   
        stars = self.starpositions  # get all of the star positions 
        x, y, z, _, scales = stars[0], stars[1], stars[2], stars[3], stars[4]
        equat, polar, radius = misc.cartesian_to_spherical(x, y, z)
        
        DGspherical = np.array([galaxy.spherical for galaxy in self.distantgalaxies])   # get all of the distant galaxy positions
        DGequat, DGpolar, DGdists = DGspherical[:, 0], DGspherical[:, 1], DGspherical[:, 2]
        # now, append the distant galaxy data to the star data since, for the sake of the program, they're functionally identical
        equat, polar, radius = np.append(equat, DGequat), np.append(polar, DGpolar), np.append(radius, DGdists)
        
        # get the radial velocities of each object
        obsvel = self.universe.radialvelocities
        DGobsvel = self.universe.distantradialvelocities
        obsvel = np.append(obsvel, DGobsvel)
        
        scales = 1 / (0.01 * np.sqrt(radius))   # determine scale much in the same way as in plot_universe() - further away objects are smaller, but this time doesnt depend on intrinsic brightness
        
        minvel = min(obsvel); maxvel = max(obsvel)
        if maxvel < -minvel:    # this conditional normalises the colourbar such that v=0 is in the middle of the max and min vel
            maxvel = -minvel
        else:
            minvel = -maxvel
    
        cm = plt.cm.get_cmap('bwr')     # blue => white => red colourmap (large negative velocities are blue - blueshifted! Nice!)
        if log:     # plot the object positions with a logarithmically scaled colourmap (looks nice, so is the default option)
            red = ax.scatter(equat, polar, c=obsvel, cmap=cm , marker='.', s=scales, linewidths=0,
                          norm=colors.SymLogNorm(linthresh=0.03, vmin=minvel, vmax=maxvel))  # note the colourmap for the redshift amount
        else:   # plots the objects with a linearly scaled colourmap - i dont recommend it, but it's here anyway
            red = ax.scatter(equat, polar, c=obsvel, vmin=minvel, vmax=maxvel, cmap=cm , marker='.', s=scales,
                             linewidths=0)  # note the colourmap for the redshift amount
        
        cbar = fig.colorbar(red, cax=cbar_ax)   # apply the colourbar to the cbar axes.
        cbar.set_label('Radial Velocity (km/s)', rotation=90)

        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')   # the background of space is black, duh
        ax.set_aspect(1)    # sets it to be twice as wide as high, so that angular ratios are preserved
        # fig.tight_layout()
        ax.invert_yaxis()
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
        
        if save == True:
            plt.close()
            return fig
        
            
    def save_data(self, properties=True, proj='Cube', pic=True, pretty=True, planetNeb=False, radio=False, stars=True, 
                  variablestars=True, blackbodies=False, distantgalax=True, flashes=True, doppler=[False, False], blackhole=False, 
                  rotcurves=False, cutoff=[True, 1e-20], site=True, archive=True):
        ''' Generates some data, takes other data, and saves it to the system in a new directory within the file directory.
        .pdf images are commented out currently - uncomment them at your own risk! (.pdfs can be in excess of 90mb each!)
        Parameters
        ----------
        properties : bool
            Whether to save the properties of the universe (e.g. variable star parameters, galaxy size, etc)
        proj : str
            One of {'AllSky', 'Cube', 'Both', 'DivCube'} to determine output of images/data
        pic : bool
            Whether to generate and save a 2d plot of the universe.
        pretty : bool
            If true, plots galaxy nebulosity on top of galaxies. Warning: high RAM usage!
        planetNeb : bool
            Whether to generate and plot planetary nebulae in the local/close galaxies
        radio : bool
            If true, plot another universe image with radio lobes overlaid.
        stars : bool
            Generate and save star data
        variablestars : bool
            Save variable star data within a subdirectory
        blackbodies : bool
            If true, plots the blackbody curve for stars in the local galaxy (1 curve for each temperature in 500K increments, so
            e.g. 1 curve for a star of temp 4500K, and another for a temp of 5000K, but not two for 4500K etc), and stores them in 
            a subdirectory under the name "{starname}-Temp:{startemp}"
        distantgalax : bool
            Generate and save distant galaxy data
        flashes : bool
            Generate and save bright event data
        doppler : list of bool
            First bool in list is whether or not to save a doppler graph with log scale. Second bool is whether to save a linear
            scaled one as well.
        blackhole : bool
            Whether to save data from black holes in all galaxies
        rotcurves : bool
            Whether to plot and save the galaxy rotation curves of all resolved galaxies. Also plots and saves galaxy cluster
            rotation curves, provided that they're resolved galaxies and the cluster has a population >= 10 galaxies.
        cutoff : list of [bool, float]
            The *minimum* flux of objects that will be 'detected' and saved into the data file. The first element is a
            bool (if True, the data is cutoff), and the second element is a float which is the minimum flux value.
        site : bool
            Whether to generate a set of .html files which mimic a crude website for a user to interact with and save data.
        '''
        print("Starting data saving..."); t0 = time()
        if not self.datadirectory:
            self.create_directory()
        if proj in ['Cube', "Both", "DivCube"] and not self.cubemapdirectory:
            method = 'Cube' if proj == "Both" else proj
            self.create_cubemap_directory(proj=method)

        # first, save the *data* that the user might interact with
        if pic:     # save a huge pic (or pics) of the universe. say goodbye to your diskspace
            self.save_pic(radio=radio, proj=proj, pretty=pretty, planetNeb=planetNeb)
        if stars:   # generate and save star data
            self.save_stars(proj=proj, cutoff=cutoff)
        if variablestars: # save variable star .txt files, and .pngs of the local galaxy variable light curves
            self.save_variables(cutoff=cutoff)
        if distantgalax:
            self.save_distant_galaxies(proj=proj, cutoff=cutoff)
        if flashes:
            self.save_flashes(proj=proj)
        if blackhole:   # save some data about black holes (radio sources)
            self.save_blackholes(proj=proj)
        # now we can save the data that will help check answers, etc
        if properties:
            self.save_properties()
            self.save_galax_properties(proj=proj)
        if blackbodies:     # plot and save blackbody curves for stars of a given temperature in the local galaxy
            self.save_blackbodies()
        if True in doppler:  # save the doppler image with a log scale
            dopplertime1 = time(); print("Saving doppler image...")
            if doppler[0]:
                fig = self.plot_doppler(save=True)
                fig.set_size_inches(18, 9, forward=True)
                fig.savefig(self.datadirectory + '/Doppler Image Log Scale.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
                # fig.savefig(self.datadirectory + '/Doppler Image Log Scale.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
            if doppler[1]:
                fig = self.plot_doppler(log=False, save=True)
                fig.set_size_inches(18, 9, forward=True)
                fig.savefig(self.datadirectory + '/Doppler Image Linear Scale.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
                # fig.savefig(self.datadirectory + '/Doppler Image Linear Scale.pdf', dpi=200, bbox_inches='tight', pad_inches = 0.01)
            dopplertime2 = time(); total = dopplertime2 - dopplertime1; print("Doppler image saved in", total, "s")
        if rotcurves:   # plot and save the rotation curve of each resolved galaxy
            self.save_rotcurves()
        if site: # create a crude html site for a user to interact with and download data
            self.create_site(directory=self.datadirectory, proj=proj, flashes=flashes, variables=variablestars)
        if archive:
            self.archive_data()
        
        t1 = time(); total = t1 - t0; print("All data generated and saved in =", total, "s")
        
    def save_properties(self):
        ''' Obtain the universe properties (physics) and write it to a file in the data directory.
        '''
        if not self.datadirectory:
            self.create_directory()

        if not self.answerdirectory:
            self.create_answerdirectory()

            
        proptime1 = time(); print("Writing universe properties...")
        # now to write the universe properties to a file. They're all pretty self-explanatory.
        text = open(self.answerdirectory + '/Universe_Details.txt', "w")
        text.write("Universe Parameters: \n")
        text.write("Parameter                   | Value   \n")
        text.write("----------------------------|---------------------------------------\n")
        text.write(f"Simulation Mode:            | {self.mode} \n")
        text.write(f"Has Black Holes?            | {self.hasblackhole} \n")
        text.write(f"Has Dark Matter?            | {self.hasdarkmatter} \n")
        text.write(f"Universe Homogeneous?       | {self.homogeneous} \n")
        text.write(f"Universe Isotropic?         | {self.isotropic} \n")
        text.write(f"Universe Radius             | {self.universe.radius} (pc)\n")
        text.write(f"Hubble Const.               | {round(self.hubble, 3)} (km/s/Mpc)\n")
        text.write(f"Local Galaxy Type           | {self.galaxies[-1].species} \n")
        text.write(f"Dist. to local galax center | {round(self.galaxies[-1].spherical[2], 2)} (pc)\n")
        text.write(f"Radius of local galaxy      | {round(self.galaxies[-1].radius, 2)} (pc)\n")
        if self.galaxies[-1].blackhole != False:
            text.write(f"Local Galax Black Hole Mass | {round(self.galaxies[-1].blackhole.mass, 2)} Solar Masses \n")
        else:
            text.write("Local Galax Black Hole Mass | N/A \n")
        text.write(f"Number of clusters          | {self.universe.clusterpop}\n")
        text.write(f"Number of galaxies          | {len(self.galaxies)} local and {len(self.distantgalaxies)} distant\n")
        text.write("\n\n")
        text.write("Variable Star Properties    | [RoughAvePeriod  LightcurveShape  PeriodLumGradient  PeriodLumYInt]\n")
        text.write("----------------------------|--------------------------------------------------------------------\n")
        for i, variable in enumerate(self.universe.variablestars):  # now to write the properties of the variable stars to file
            if i == 0:
                text.write(f"Variable Stars?             | {variable}\n")
            else:
                length = "Short" if i == 1 else "Long"
                length = "Longest" if i == 3 else length
                text.write(f"Variable Class:             | {length}\n")
                text.write(f"                            | {variable}\n")
                
        text.close()
        print('Universe properties saved to', self.answerdirectory + '/Universe_Details.txt')
        # plot and save the hubble diagram for the universe
        hubblediag = self.universe.plot_hubblediagram(save=True) 
        hubblediag.savefig(self.datadirectory + '/Hubble_Diagram.png', dpi=600, bbox_inches='tight', pad_inches = 0.01)
        # hubblediag.savefig(self.datadirectory + '/Hubble Diagram.pdf', dpi=600, bbox_inches='tight', pad_inches = 0.01)
        # now plot and save the HR diagram for the local galaxy
        HR = self.galaxies[-1].plot_HR(isoradii=True, xunit="both", yunit="BolLumMag", variable=True, save=True)
        HR.savefig(self.datadirectory + '/Local_Galaxy_HR_Diagram.png', dpi=600, bbox_inches='tight', pad_inches = 0.01)
        plt.close('all')
        proptime2 = time(); total = proptime2 - proptime1; print("Universe properties saved in", total, "s")
        
    def save_galax_properties(self, proj="Cube"):
        ''' Saves two .txt files containing important properties about all of the resolved galaxies, and galaxy 
        clusters (resolved and distant) respectively.
        Parameters
        ----------
        proj : str
            One of {'AllSky', 'Cube', 'Both', 'DivCube'} depending on if you want a single, equirectangular projected image of the sky, 
            or six, cubemapped images of the sky, respectively. (or both!)
        '''
        if not self.datadirectory:
            self.create_directory()
        
        # first, save a file with all galaxy data (with formatted values)
        with open(self.answerdirectory + "/Galaxy_Details.txt", "w") as galax:
            if proj in ["Cube", "DivCube", "Both"]:
                directions = np.array(["Front", "Back", "Top", "Bottom", "Left", "Right"])
                galax.write("Face X Y Equat Polar Type Distance(pc) Radius(pc) NumStars Mass(Mo) BHMass(Mo)\n")
                for galaxy in self.galaxies:
                    x, y, z = galaxy.cartesian
                    [uc], [vc], [index] = misc.cubemap(x, y, z)
                    face = directions[index]
                    equat, polar, dist = galaxy.spherical
                    uc, vc = format(uc, '2.2f'), format(vc, '2.2f')
                    equat, polar, dist = format(equat, '3.2f'), format(polar, '3.2f'), format(dist, '.0f')
                    radius, mass, bhmass = format(galaxy.radius, '.0f'), format(galaxy.galaxymass, '.0f'), format(galaxy.galaxyBHmass, '.0f')
                    galax.write(f"{face} {uc} {vc} {equat} {polar} {galaxy.species} {dist} {radius} {len(galaxy.stars)} {mass} {bhmass}\n")
            else:
                galax.write("Equat Polar Type Distance(pc) Radius(pc) NumStars Mass(Mo) BHMass(Mo)\n")
                for galaxy in self.galaxies:
                    equat, polar, dist = galaxy.spherical
                    equat, polar, dist = format(equat, '3.2f'), format(polar, '3.2f'), format(dist, '.0f')
                    radius, mass, bhmass = format(galaxy.radius, '.0f'), format(galaxy.galaxymass, '.0f'), format(galaxy.galaxyBHmass, '.0f')
                    galax.write(f"{equat} {polar} {galaxy.species} {dist} {radius} {len(galaxy.stars)} {mass} {bhmass}\n")
        print('Galaxy properties saved to', self.answerdirectory + "/Galaxy_Details.txt")
                    
        # now save a file with all cluster data (formatted values)
        with open(self.answerdirectory + "/Cluster_Details.txt", "w") as clust:
            if proj in ["Cube", "DivCube", "Both"]:
                directions = np.array(["Front", "Back", "Top", "Bottom", "Left", "Right"])
                clust.write("Face X Y Equat Polar NumGalax Distance(pc) Radius(pc) Mass(Mo)\n")
                for cluster in self.universe.clusters:
                    x, y, z = cluster.cartesian
                    [uc], [vc], [index] = misc.cubemap(x, y, z)
                    face = directions[index]
                    equat, polar, dist = cluster.spherical
                    uc, vc = format(uc, '2.2f'), format(vc, '2.2f')
                    equat, polar, dist = format(equat, '3.2f'), format(polar, '3.2f'), format(dist, '.0f')
                    radius, mass = format(cluster.radius, '.0f'), format(cluster.clustermass, '.0f')
                    clust.write(f"{face} {uc} {vc} {equat} {polar} {cluster.clusterpop} {dist} {radius} {mass}\n")
            else:
                clust.write("Equat Polar NumGalax Distance(pc) Radius(pc) Mass(Mo)\n")
                for cluster in self.universe.clusters:
                    equat, polar, dist = cluster.spherical
                    equat, polar, dist = format(equat, '3.2f'), format(polar, '3.2f'), format(dist, '.0f')
                    radius, mass = format(cluster.radius, '.0f'), format(cluster.clustermass, '.0f')
                    clust.write(f"{equat} {polar} {cluster.clusterpop} {dist} {radius} {mass}\n")
        print('Cluster properties saved to', self.answerdirectory + "/Cluster_Details.txt")
            
        
    def save_pic(self, radio=False, proj='AllSky', pretty=True, planetNeb=False):
        ''' Saves the universe picture(s) in the directory of choice, given desired projection and radio contours.
        Parameters
        ----------
        radio : bool
            True if you want an *extra* image with radio contours overlaid onto the AllSky projection
        proj : str
            One of {'AllSky', 'Cube', 'Both', 'DivCube'} depending on if you want a single, equirectangular projected image of the sky, 
            or six, cubemapped images of the sky, respectively. (or both!)
        pretty : bool
            Whether to plot galaxy nebulosity on top of galaxies. Warning: high RAM usage!
        planetNeb : bool
            Whether to generate and plot planetary nebulae in the local/close galaxies
        '''
        # this import and backend change apparently helps with time to save images 
        import matplotlib
        matplotlib.use('Agg') # backend which doesn't produce new windows for each figure. helps when producing/saving lots of images
        
        if not self.datadirectory: # create the directory if it doesn't yet exist
            self.create_directory()
        if proj in ['Cube', "Both", "DivCube"] and not self.cubemapdirectory: # create the cubemap directory if it doesnt yet exist (for cubemapped datasets)
            method = 'Cube' if proj == "Both" else proj
            self.create_cubemap_directory(proj=method)
            
        # now to save cubemapped if desired
        if proj in ['Cube', 'Both', 'DivCube']:
            pictime1 = time(); print("Generating universe picture...") # start a timer
            figAxes = self.plot_universe(pretty=pretty, planetNeb=planetNeb, save=True, cubemap=True) # plot the universe
            directions = ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right'] # 6 cubemap directions
            for i in range(6):
                # for each direction, we're gonna save a big picture
                fig, ax = figAxes[i] # get the figure corresponding to the ith face
                if proj != 'DivCube':
                    fig.savefig(self.datadirectory + f'/{directions[i]}/{directions[i]}.png', dpi=1500, bbox_inches='tight', 
                            pad_inches = 0.01)
                
                if proj == 'DivCube':
                    fig.set_size_inches(1, 1)
                    # in each direction, now save 36 subdivisions of the face (making a total of 6 * 36 subdivisions)
                    for j, X in enumerate(["A", "B", "C", "D", "E", "F"]):
                        xbounds = [-45 + j * 15, -30 + j * 15] # calculate the xbounds of this subdivision
                        ax.set_xlim(xbounds) # set xbounds
                        for Y in range(1, 7):
                            ybounds = [45 - Y * 15, 60 - Y * 15] # calculate ybounds of this subdivision
                            ax.set_ylim(ybounds) # set ybounds
                            # now to save this particular subdivision image
                            fig.savefig(self.datadirectory + f'/{directions[i]}/{X}{Y}/{X}{Y}_{directions[i]}.png', 
                                        dpi=750, bbox_inches='tight', pad_inches = 0.01)
                            
                    # now to save a picture of the entire face, with grid lines indicating subdivisions
                    fig.set_size_inches(6, 6)
                    ax.set_ylim(-45, 45); ax.set_xlim(-45, 45)
                    ax.grid(visible=False)
                    for loc in [-30, -15, 0, 15, 30]:
                        ax.axhline(y=loc, lw=0.2, c='w'); ax.axvline(x=loc, lw=0.2, c='w')
                    for j, col in enumerate(["A", "B", "C", "D", "E", "F"]):
                        ax.text(-40 + 15*j, 40, col, c='w', fontsize='large')
                    for j, row in enumerate(["1", "2", "3", "4", "5", "6"]):
                        ax.text(-44, 32 - j*15, row, c='w', fontsize='large')
                    fig.savefig(self.datadirectory + f'/{directions[i]}/{directions[i]}.png', dpi=1500, bbox_inches='tight', 
                            pad_inches = 0.01)
                
            pictime2 = time(); total = pictime2 - pictime1; print("Cubemapped universe pictures saved in", total, "s")
            
        # now to plot an equirectangular image of the universe if desired
        if proj in ['AllSky', 'Both']:
            pictime1 = time(); print("Generating universe picture...") # start timer
            fig, ax = self.plot_universe(pretty=pretty, planetNeb=planetNeb, save=True)
            fig.set_size_inches(18, 9, forward=True)
            fig.savefig(self.datadirectory + '/AllSky_Universe_Image.png', dpi=1500, bbox_inches='tight', 
                        pad_inches = 0.01)
            pictime2 = time(); total = pictime2 - pictime1; print("AllSky Universe picture saved in", total, "s")
        
        if radio and self.hasblackhole:       # plot radio data too
            print("Generating radio overlay...")
            if proj in ["AllSky", "Both"]:
                self.plot_radio([fig, ax], method="AllSky")
                fig.savefig(self.datadirectory + '/AllSky_Radio_Overlay_Image.png', dpi=1500, bbox_inches='tight', 
                            pad_inches = 0.01)
                pictime3 = time(); total = pictime3 - pictime2; print("Radio overlay picture saved in", total, "s")
            if proj in ["Cube", "Both", "DivCube"]:
                self.plot_radio(figAxes, method="Cube")
                for i in range(6):
                    fig, ax = figAxes[i]
                    ax.set_xlim(-45, 45); ax.set_ylim(-45, 45);
                    fig.savefig(self.datadirectory + f'/{directions[i]}/{directions[i]}_Radio_Overlay.png', dpi=1500, 
                                bbox_inches='tight', pad_inches = 0.01)
                    if proj == 'DivCube':
                        for j, X in enumerate(["A", "B", "C", "D", "E", "F"]):
                            xbounds = [-45 + j * 15, -30 + j * 15]
                            ax.set_xlim(xbounds)
                            for Y in range(1, 7):
                                ybounds = [45 - Y * 15, 60 - Y * 15]
                                ax.set_ylim(ybounds)
                                fig.savefig(self.datadirectory + f'/{directions[i]}/{X}{Y}/{X}{Y}_{directions[i]}_Radio_Overlay.png',
                                            dpi=150, bbox_inches='tight', pad_inches = 0.01)
            
        plt.close('all')
        
            
    def save_stars(self, proj='AllSky', cutoff=[True, 1e-22]):
        ''' Saves the data of all of the stars in the data directory, based on the projection of the images.
        Parameters
        ----------
        proj : str
            One of {'AllSky', 'Cube', 'Both', 'DivCube'} depending on if you want a single, equirectangular projected image of the sky, or
            six, cubemapped images of the sky, respectively. Both is also an option. 
        cutoff : list of [bool, float]
            The *minimum* flux of objects that will be 'detected' and saved into the data file. The first element is a
            bool (if True, the data is cutoff), and the second element is a float which is the minimum flux value.
        '''
        if not self.datadirectory:
            self.create_directory()
        if proj in ['Cube', "Both", "DivCube"] and not self.cubemapdirectory:
            self.create_cubemap_directory()
            
        startime1 = time(); print("Generating star data...")
        #firstly, get star xyz positions and convert them to equatorial/polar
        starpos = self.starpositions    
        x, y, z, _, _ = starpos[0], starpos[1], starpos[2], starpos[3], starpos[4]
        
        if proj in ["Cube", "Both", "DivCube"]:
            uc, vc, index = misc.cubemap(x, y, z)
            if proj in ["Cube", "DivCube"]:
                _, _, radius = misc.cartesian_to_spherical(x, y, z)
        if proj in ['AllSky', 'Both']:
            equat, polar, radius = misc.cartesian_to_spherical(x, y, z)
            
        # generate parallax according to p = 1/d (pc) formula, with +/- 0.5 arcsecond of uncertainty (to keep within +/-2mas for objects with no inherent parallax)
        parallax = (1 / radius) + np.random.normal(0, 0.0005, self.numstars)    
        parallax = np.around(parallax, decimals=3)
        
        # round position values to 4 decimal places which is about ~1/3 of an arcsecond (or rather 0.0001 degrees)
        if proj in ["AllSky", "Both"]:
            # equat = np.around(equat, decimals=4); polar = np.around(polar, decimals=4);
            equat = np.array([format(coord, '3.4f') for coord in equat]); polar = np.array([format(coord, '3.4f') for coord in polar])
        if proj in ["Cube", "Both", "DivCube"]:
            # uc = np.around(uc, decimals=4); vc = np.around(vc, decimals=4);
            if proj == "DivCube": # we need to keep the numerical vals (opposed to str vals) to do conditionals later on
                num_uc = uc; num_vc = vc;
            uc = np.array([format(coord, '2.4f') for coord in uc]); vc = np.array([format(coord, '2.4f') for coord in vc])
            
        # now to work with the band luminosity data. First, get the data for each star in the universe
        blueflux = [[star.bandlumin[0] for star in galaxy.stars] for galaxy in self.galaxies]
        greenflux = [[star.bandlumin[1] for star in galaxy.stars] for galaxy in self.galaxies]
        redflux = [[star.bandlumin[2] for star in galaxy.stars] for galaxy in self.galaxies]
        # now, flatten the above arrays and divide them by the distance to the star, squared [for SI unit conversion]
        blueflux = np.array([flux for galaxy in blueflux for flux in galaxy]) / (4 * np.pi * (3.086 * 10**16 * radius)**2)
        greenflux = np.array([flux for galaxy in greenflux for flux in galaxy]) / (4 * np.pi * (3.086 * 10**16 * radius)**2)
        redflux = np.array([flux for galaxy in redflux for flux in galaxy]) / (4 * np.pi * (3.086 * 10**16 * radius)**2)
        obsvel = self.universe.radialvelocities     # retrieve the radial velocities from the universe object
        obsvel = np.around(obsvel, decimals=2)
        
        # now to append data to the star.txt file as to whether a particular star shows a variable light curve
        variabool = np.zeros(self.numstars)    # get it? variable but in terms of bools
        k = 0
        for galaxy in self.galaxies:
            for star in galaxy.stars:
                variabool[k] = star.variable
                k += 1
        variabool = variabool.astype('int32')   # convert those floats to integers!
        
        names = self.starnames
        if cutoff[0] == True:
            validIndexes = [i for i in range(len(blueflux)) if np.min(np.array([blueflux[i], greenflux[i], redflux[i]])) > cutoff[1]]
            
            if proj in ["Cube", "Both", "DivCube"]:
                uc, vc, index = uc[validIndexes], vc[validIndexes], index[validIndexes]
                if proj == "DivCube":
                    num_uc = num_uc[validIndexes]; num_vc = num_vc[validIndexes]
            if proj in ['AllSky', 'Both']:
                equat, polar = equat[validIndexes], polar[validIndexes]
            redflux, blueflux, greenflux = redflux[validIndexes], blueflux[validIndexes], greenflux[validIndexes]
            parallax, obsvel, variabool = parallax[validIndexes], obsvel[validIndexes], variabool[validIndexes]
            names = names[validIndexes]
        
        blueflux = np.array([format(flux, '.3e') for flux in blueflux])   # now round each data point to 3 decimal places
        greenflux = np.array([format(flux, '.3e') for flux in greenflux])
        redflux = np.array([format(flux, '.3e') for flux in redflux])
        
        if proj in ["Cube", "Both", "DivCube"]:
            directions = ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right']
            for i, direction in enumerate(directions):
                # now, write all star data to a pandas dataframe
                
                if proj in ["Cube", "Both"]:
                    dirNames = [f"{direction}{name}" for name in names[index == i]]
                    stardata = {'Name': dirNames, 
                                'X': uc[index == i], 'Y': vc[index == i],        # units of the X/Y are in degrees
                                'BlueF': blueflux[index == i], 'GreenF': greenflux[index == i], 'RedF': redflux[index == i],   # units of these fluxes are in W/m^2/nm
                                'Parallax': parallax[index == i], 'RadialVelocity': obsvel[index == i], # units of parallax are in arcsec, obsvel in km/s
                                'Variable?': variabool[index == i]}  # outputs 1.0 if variable, 0.0 if not      
                    starfile = pd.DataFrame(stardata)
                    
                    starfile.to_csv(self.datadirectory + f"/{direction}/Star_Data.csv", index=None, sep=',')    # and finally save the dataframe to the directory
                elif proj == "DivCube":
                    for j, X in enumerate(["A", "B", "C", "D", "E", "F"]):
                        xbounds = [-45 + j * 15, -30 + j * 15]
                        for Y in range(1, 7):
                            ybounds = [45 - Y * 15, 60 - Y * 15]
                            indices = np.where((index == i) & (num_uc >= xbounds[0]) & (num_uc <= xbounds[1]) &
                                               (num_vc >= ybounds[0]) & (num_vc <= ybounds[1]))
                            divNames = [f"{direction}{name}" for name in names[indices]]
                            stardata = {'Name': divNames, 
                                        'X': uc[indices], 'Y': vc[indices],        # units of the X/Y are in degrees
                                        'BlueF': blueflux[indices], 'GreenF': greenflux[indices], 'RedF': redflux[indices],   # units of these fluxes are in W/m^2/nm
                                        'Parallax': parallax[indices], 'RadialVelocity': obsvel[indices], # units of parallax are in arcsec, obsvel in km/s
                                        'Variable?': variabool[indices]}  # outputs 1.0 if variable, 0.0 if not      
                            starfile = pd.DataFrame(stardata)
                            
                            starfile.to_csv(self.datadirectory + f"/{direction}/{X}{Y}/Star_Data.csv", index=None, sep=',')    # and finally save the dataframe to the directory
                            print('DivCube star data saved to',self.datadirectory + f"/{direction}/{X}{Y}/Star_Data.csv")
        if proj in ["AllSky", "Both"]:
            # now, write all star data to a pandas dataframe
            stardata = {'Name': names, 'Equatorial': equat, 'Polar': polar,        # units of the equat/polar are in degrees
                        'BlueF': blueflux, 'GreenF': greenflux, 'RedF': redflux,   # units of these fluxes are in W/m^2/nm
                        'Parallax': parallax, 'RadialVelocity': obsvel,           # units of parallax are in arcsec, obsvel in km/s
                        'Variable?': variabool}                                  # outputs 1.0 if variable, 0.0 if not      
            starfile = pd.DataFrame(stardata)
            
            starfile.to_csv(self.datadirectory + "/All_Star_Data.csv", index=None, sep=',')    # and finally save the dataframe to the directory
            print('Star data saved to',self.datadirectory + "/All_Star_Data.csv")
        startime2 = time(); total = startime2 - startime1; print("Star Data saved in", total, "s")
        
    def save_distant_galaxies(self, proj='AllSky', evolution=True, cutoff=[True, 1e-22]):
        ''' Saves the data of all of the distant galaxies in the data directory, based on the projection of the images.
        Parameters
        ----------
        proj : str
            One of {'AllSky', 'Cube', 'Both', 'DivCube'} depending on if you want a single, equirectangular projected image of the sky, or
            six, cubemapped images of the sky, respectively. Both is also an option.
        evolution : bool
            If True, galaxies in the distant (young) universe will have their spectra blueshifted so as to emulate younger
            galaxies being bluer than their more evolved counterparts. 
        cutoff : list of [bool, float]
            The *minimum* flux of objects that will be 'detected' and saved into the data file. The first element is a
            bool (if True, the data is cutoff), and the second element is a float which is the minimum flux value.
        '''
        distanttime1 = time(); print("Saving distant galaxy data...")
        sphericals = np.array([galaxy.spherical for galaxy in self.distantgalaxies])
        if proj in ["AllSky", "Both"]:
            equat, polar, dists = sphericals[:, 0], sphericals[:, 1], sphericals[:, 2]
        if proj in ["Cube", "Both", "DivCube"]:
            _, _, dists = sphericals[:, 0], sphericals[:, 1], sphericals[:, 2]
            carts = np.array([galaxy.cartesian for galaxy in self.distantgalaxies])
            x, y, z = carts[:, 0], carts[:, 1], carts[:, 2]
            uc, vc, index = misc.cubemap(x, y, z)
        distsMeters = dists * 3.086 * 10**16
        
        bandlumin = np.array([galaxy.get_bandlumin(evolution=evolution, universe_rad=self.universe.radius) for galaxy in self.distantgalaxies])
        bluelumin, greenlumin, redlumin = bandlumin[:, 0], bandlumin[:, 1], bandlumin[:, 2]
        blueflux, greenflux, redflux = bluelumin / (4*np.pi*distsMeters**2), greenlumin / (4*np.pi*distsMeters**2), redlumin / (4*np.pi*distsMeters**2) 
        
        radii = np.array([galaxy.radius for galaxy in self.distantgalaxies])
        sizes = 2 * np.arctan((radii / 2) / dists)      # gets the apparent size of the galaxy (thanks trig!)
        sizes = np.rad2deg(sizes) * 3600    # this gives the size of the galaxy in units of arcseconds
        sizes = np.around(sizes, decimals=4)
        
        DGobsvel = self.universe.distantradialvelocities
        DGobsvel = np.around(DGobsvel, decimals=2)
        
        width = int(-(-np.log10(len(self.distantgalaxies)) // 1))
        names = np.array([f"DG{i:0{width}d}" for i in range(1, len(self.distantgalaxies) + 1)])
            
        if cutoff[0] == True:
            validIndexes = [i for i in range(len(blueflux)) if np.min(np.array([blueflux[i], greenflux[i], redflux[i]])) > cutoff[1]]
            
            if proj in ["Cube", "Both", "DivCube"]:
                uc, vc, index = uc[validIndexes], vc[validIndexes], index[validIndexes]
            if proj in ['AllSky', 'Both']:
                equat, polar = equat[validIndexes], polar[validIndexes]
            redflux, blueflux, greenflux = redflux[validIndexes], blueflux[validIndexes], greenflux[validIndexes]
            sizes, DGobsvel, names = sizes[validIndexes], DGobsvel[validIndexes], names[validIndexes]
            
        # now to format the data so that it's clean and pretty in the .txt file
        if proj in ["AllSky", "Both"]:
            equat = [format(coord, '3.4f') for coord in equat]; polar = [format(coord, '3.4f') for coord in polar]
        if proj in ["Cube", "Both", "DivCube"]:
            if proj == "DivCube": # we need to keep the numerical vals (opposed to str vals) to do conditionals later on
                num_uc = uc; num_vc = vc;
            uc = np.array([format(coord, '2.4f') for coord in uc]); vc = np.array([format(coord, '2.4f') for coord in vc])
        blueflux = np.array([format(flux, '.3e') for flux in blueflux])   # now round each data point to 3 decimal places
        greenflux = np.array([format(flux, '.3e') for flux in greenflux])
        redflux = np.array([format(flux, '.3e') for flux in redflux])
        
        if proj in ["AllSky", "Both"]:
            DGdata = {"Name": names, 'Equatorial': equat, 'Polar': polar,
                      'BlueF': blueflux, 'GreenF': greenflux, 'RedF': redflux,
                      'Size': sizes, 'RadialVelocity': DGobsvel}
            DGfile = pd.DataFrame(DGdata)
            DGfile.to_csv(self.datadirectory + "/Distant_Galaxy_Data.csv", index=None, sep=',')
            
        if proj in ["Cube", "Both", "DivCube"]:
            directions = ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right']
            for i, direction in enumerate(directions):
                if proj in ["Cube", "Both"]:
                    dirNames = [f"{direction}{name}" for name in names[index == i]]
                    DGdata = {"Name": dirNames, 'X': uc[index == i], 'Y': vc[index == i],
                              'BlueF': blueflux[index == i], 'GreenF': greenflux[index == i], 'RedF': redflux[index == i],
                              'Size': sizes[index == i], 'RadialVelocity': DGobsvel[index == i]}
                    DGfile = pd.DataFrame(DGdata)
                    fname = self.datadirectory + f"/{direction}/Distant_Galaxy_Data.csv"
                    DGfile.to_csv(fname, index=None, sep=',')
                elif proj == "DivCube":
                    for j, X in enumerate(["A", "B", "C", "D", "E", "F"]):
                        xbounds = [-45 + j * 15, -30 + j * 15]
                        for Y in range(1, 7):
                            ybounds = [45 - Y * 15, 60 - Y * 15]
                            indices = np.where((index == i) & (num_uc >= xbounds[0]) & (num_uc <= xbounds[1]) &
                                               (num_vc >= ybounds[0]) & (num_vc <= ybounds[1]))
                            divNames = [f"{direction}{name}" for name in names[indices]]
                            DGdata = {"Name": divNames, 'X': uc[indices], 'Y': vc[indices],
                                      'BlueF': blueflux[indices], 'GreenF': greenflux[indices], 'RedF': redflux[indices],
                                      'Size': sizes[indices], 'RadialVelocity': DGobsvel[indices]}    
                            DGfile = pd.DataFrame(DGdata)
                            fname = self.datadirectory + f"/{direction}/{X}{Y}/Distant_Galaxy_Data.csv"
                            DGfile.to_csv(fname, index=None, sep=',')    # and finally save the dataframe to the directory
        print('Saved to', fname)
        distanttime2 = time(); total = distanttime2 - distanttime1; print("Distant galaxy data saved in", total, "s")
        
    def save_variables(self, cutoff=[True, 1e-22]):
        ''' Saves .txt files of variable star data for close stars in the universe, and images for variable light curves in the local
            galaxy.
        Parameters
        ----------
        cutoff : list of [bool, float]
            The *minimum* flux of objects that will be 'detected' and saved into the data file. The first element is a
            bool (if True, the data is cutoff), and the second element is a float which is the minimum flux value.
        '''
        vartime1 = time(); print("Saving variable data...")
        variabledirectory = self.datadirectory + "/Variable_Star_Data"     # save data within a subfolder
        os.makedirs(variabledirectory)
        k = 0
        
        starpos = self.starpositions    
        x, y, z, _, _ = starpos[0], starpos[1], starpos[2], starpos[3], starpos[4]
        _, _, radius = misc.cartesian_to_spherical(x, y, z)
        
        directions = ['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right']
        
        for galaxy in self.galaxies:
            for star in galaxy.stars:
                if star.variable == True:
                    starflux = star.bandlumin / (4 * np.pi * (radius[k] * 3.086 * 10**16)**2) # calculate flux of this variable star
                    if (min(starflux) > cutoff[1]) or not cutoff[0]: # if the min bandflux is greater than the cutoff OR if we dont care about cutoffs
                        starname = self.starnames[k]
                        
                        X, Y, Z = np.array(x[k]), np.array(y[k]), np.array(z[k])
                        _, _, index = misc.cubemap(X, Y, Z)
                        direction = directions[index[0]]
                        
                        if galaxy.rotate == False:      # must be the local galaxy, so we want to save a pic of the lightcurve
                            fig = star.plot_lightcurve(save=True)
                            fig.savefig(variabledirectory + f'/{direction}{starname}.png', dpi=400, bbox_inches='tight', pad_inches = 0.01)
                        times, fluxes = star.lightcurve
                        variabledata = {"Time": times, "NormalisedFlux": fluxes}
                        variablefile = pd.DataFrame(variabledata)
                        fname = variabledirectory + f"/{direction}{starname}.csv"
                        variablefile.to_csv(fname, index=None, sep=',')
                        print('Saved to', fname)
                k += 1
        
        # now to take and plot the period-luminosity data of the local galaxy!
        periods, lumins = [], []
        for star in self.galaxies[-1].stars:
            if star.variable == True:
                periods.append(star.period)
                lumins.append(star.luminosity)
        fig, ax = plt.subplots()
        ax.scatter(periods, lumins, s=0.5)
        ax.set_yscale('log'); ax.set_xlim(xmin=0)
        ax.set_xlabel("Period (hours)"); ax.set_ylabel(r"Log Luminosity ($L / L_\odot$)")
        plt.close()
        fig.savefig(self.datadirectory + '/Period_Luminosity_Data.png', dpi=400, bbox_inches='tight', pad_inches = 0.01)
        
        vartime2 = time(); total = vartime2 - vartime1; print("Variable data saved in", total, "s")
    
    def save_flashes(self, proj='AllSky'):
        ''' Saves all of the bright flash data into the data directory, with coordinates of the events based on the projection.
        Parameters
        ----------
        proj : str
            The coordinate system for the bright flash events. One of {'AllSky', 'Cube', 'Both'}. If 'Cube' is chosen, the direction
            is output into the file too, with corresponding local coordinates.
        '''
        supertime1 = time(); print("Saving bright flash data...")
        pos, peak = self.flashes
        
        if proj in ["AllSky", "Both"]:
            equats = np.array([format(abs(equat), '3.2f') for equat in pos[0]])
            polars = np.array([format(abs(polar), '3.2f') for polar in pos[1]])
        if proj in ["Cube", "Both", "DivCube"]:
            x, y, z = misc.spherical_to_cartesian(pos[0], pos[1], pos[2])
            uc, vc, index = misc.cubemap(x, y, z)
            uc = np.array([format(coord, '3.2f') for coord in uc])
            vc = np.array([format(coord, '3.2f') for coord in vc])
        
        width = int(-(-np.log10(len(peak)) // 1))
        if self.eventType == "supernova":
            peak = np.array([format(flux, '.3e') for flux in peak])
            names = [f"SNe{i:0{width}d}" for i in range(1, len(peak) + 1)]
        elif self.eventType == "flash":
            names = [f"FE{i:0{width}d}" for i in range(1, len(peak) + 1)]
        
        if proj in ["AllSky", "Both"]:
            if self.eventType == "supernova":
                flashdata = {"Name": names, "Equatorial": equats, "Polar": polars, "PeakFlux(W/m^2)": peak}
            elif self.eventType == "flash":
                flashdata = {"Name": names, "Equatorial": equats, "Polar": polars, "Photon-Count": peak}
            flashfile = pd.DataFrame(flashdata)
            fname = self.datadirectory + "/AllSky_Flash_Data.csv"
            flashfile.to_csv(fname, index=None, sep=',')
        if proj in ["Cube", "Both", "DivCube"]:
            directions = np.array(['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right'])
            if self.eventType == "supernova":
                flashdata = {"Name": names, "Direction": directions[index], "X": uc, "Y": vc, "PeakFlux(W/m^2)": peak}
            elif self.eventType == "flash":
                flashdata = {"Name": names, "Direction": directions[index], "X": uc, "Y": vc, "Photon-Count": peak}
            flashfile = pd.DataFrame(flashdata)
            fname = self.datadirectory + "/Flash_Data.csv"
            flashfile.to_csv(fname, index=None, sep=',')
        print('Saved to', fname)
        supertime2 = time(); total = supertime2 - supertime1; print("Flash data saved in", total, "s")
    
    def save_blackbodies(self):
        ''' Saves blackbody curves for all stars in the local galaxy. 
        '''
        blacktime1 = time(); print("Generating blackbody curves for local galaxy...")
        blackbodydirectory = self.datadirectory + "/Local_Galaxy_Blackbody_Curves"     # save data within a subfolder
        os.makedirs(blackbodydirectory)
        k = 0
        blackbodytemps = np.arange(0, 50)     # we only want two curves for each 1000K temps, so these are the temps/500
        for galaxy in self.galaxies:
            if galaxy.rotate == False:  # must be the local galaxy, so lets plot some blackbody curves!
                for i, star in enumerate(galaxy.stars):
                    roundtemp = round(star.temperature/500)
                    if roundtemp in blackbodytemps:
                        fig = star.plot_blackbodycurve(markers=True, visible=True, save=True)
                        fig.savefig(blackbodydirectory + f'/{self.starnames[k]}-Temp={round(star.temperature)}.png', 
                                    dpi=400, bbox_inches='tight', pad_inches = 0.01)
                        blackbodytemps = np.where(blackbodytemps==roundtemp, 0, blackbodytemps)     # remove this temperature from the pool of temps to plot
                    k += 1
            else:   # we still need to increment the ticker so that later data is accurate
                k += len(galaxy.stars)
        plt.close('all')
        blacktime2 = time(); total = blacktime2 - blacktime1; print("Blackbody curves saved in", total, "s")
        
    def save_blackholes(self, proj='AllSky'):
        ''' Saves all of the black hole luminosity data into the data directory, with coordinates based on the projection.
        Parameters
        ----------
        proj : str
            The coordinate system for the black holes. One of {'AllSky', 'Cube', 'Both'}. If 'Cube' is chosen, the direction
            is output into the file too, with corresponding local coordinates.
        '''
        if not self.blackholes:
            return 0
        
        bhtime1 = time(); print("Saving black hole data...")
        equats, polars, dists, BHlumins = [], [], [], []
        for galaxy in self.allgalaxies:
            if galaxy.blackhole != False:
                equat, polar, dist = galaxy.spherical
                distM = dist * 3.086 * 10**16     # get the distance to the BH in meters
                lumin = (galaxy.blackhole.luminosity * 3.828 * 10**26) / distM**2    # get the lumin in W/m^2
                if lumin >= 10**-17:        # we set a hard limit on the distance we can detect black holes
                    BHlumins.append(lumin); equats.append(equat); polars.append(polar); dists.append(dist)
        
        equats = np.array([format(abs(equat), '3.2f') for equat in equats])
        polars = np.array([format(abs(polar), '3.2f') for polar in polars])
        BHlumins = np.array([format(flux, '.3e') for flux in BHlumins])
        width = int(-(-np.log10(len(BHlumins)) // 1))
        names = [f"RS{i:0{width}d}" for i in range(1, len(BHlumins) + 1)]
        if proj in ["AllSky", "Both"]:
            BHdata = {'Name': names, 'Equatorial': equats, 'Polar': polars,
                      'Luminosity': BHlumins}
            BHfile = pd.DataFrame(BHdata)
            fname = self.datadirectory + "/AllSky_Radio_Source_Data.csv"
            BHfile.to_csv(fname, index=None, sep=',')
        if proj in ["Cube", "Both", "DivCube"]:
            x, y, z = misc.spherical_to_cartesian(equats, polars, np.array(dists))
            uc, vc, index = misc.cubemap(x, y, z)
            
            directions = np.array(['Front', 'Back', 'Top', 'Bottom', 'Left', 'Right'])
            
            BHdata = {'Name': names, 'Direction': directions[index], 'X': uc, 'Y': vc,
                      'Luminosity': BHlumins}
            BHfile = pd.DataFrame(BHdata)
            fname = self.datadirectory + "/Radio_Source_Data.csv"
            BHfile.to_csv(fname, index=None, sep=',')
        print('Saved to',fname)
        bhtime2 = time(); total = bhtime2 - bhtime1; print("Black hole data saved in", total, "s")
    
    def save_rotcurves(self):
        ''' Save rotation curves for all of the galaxies and clusters in the universe, each in their own file. 
        '''
        rottime1 = time(); print("Saving galaxy rotation curves...")
        rotcurvedirectory = self.answerdirectory + "/Galaxy_Rotation_Curves"
        os.makedirs(rotcurvedirectory)  # make a new folder to hold the rotation curves
        
        for galaxy in self.galaxies:
            equat, polar, _ = galaxy.spherical; equat, polar = round(equat, 2), round(polar, 2)
            bh = "1" if galaxy.blackhole != False else "0"
            dm = "1" if galaxy.darkmatter == True else "0"
            fig = galaxy.plot_RotCurve(newtapprox=True, save=True)
            fig.savefig(rotcurvedirectory + f'/E{equat}_P{polar}_{galaxy.species}_BH{bh}_DM{dm}.png', 
                        dpi=200, bbox_inches='tight', pad_inches = 0.01)
        rottime2 = time(); total = rottime2 - rottime1; print("Galaxy rotation curves saved in", total, "s")
        
        print("Saving cluster rotation curves...")
        clustercurvedirectory = self.answerdirectory + "/Cluster_Rotation_Curves"
        os.makedirs(clustercurvedirectory)
        for cluster in self.universe.clusters:
            pop = cluster.clusterpop
            if pop >= 10 and cluster.complexity != "Distant":
                equat, polar, _ = cluster.spherical; equat, polar = round(equat, 2), round(polar, 2)
                fig = cluster.plot_RotCurve(newtapprox=True, save=True)
                fig.savefig(clustercurvedirectory + f'/E{equat}_P{polar}_Pop;{pop}.png', 
                            dpi=400, bbox_inches='tight', pad_inches = 0.01)
        plt.close('all')
        rottime3 = time(); total = rottime3 - rottime2; print("Cluster rotation curves saved in", total, "s")
    
    def create_site(self, directory, proj, flashes=True, variables=True):
        ''' Creates a set of .html files which mimics a crude website to interact with and save the data. 
        Parameters
        ----------
        directory : str
            The main data directory of this UniverseSim dataset
        proj : str
            One of {'AllSky', 'Cube', 'Both', 'DivCube'} to determine output of images/data
        flashes : bool
            If we've saved the flash events. Will generate a section in the top-level index file about xray flashes
        '''
        print("Generating HTML files...")
        t1 = time()
        Site(directory, proj=proj, flashes=flashes, variables=variables)
        t2 = time(); t = t2 - t1; print(f"HTML Files created in {t} s")
        
    def archive_data(self):
        ''' Adds all of the files from *this* dataset to a .zip archive (with the same name) in the Datasets folder.
        '''
        print("Archiving data to .zip file...")
        t1 = time()
        import shutil
        shutil.make_archive(self.datadirectory, 'zip', self.datadirectory + "/")
        t2 = time(); t = t2 - t1; print(f"Data archived in {t} s")
        
def main():
    ap = ArgumentParser(description='Generate realistic simulated universes for PHYS3080 at UQ')
    ap.add_argument('-s', '--seed', metavar='S', type=int, help='Random Seed', default=6683)
    ap.add_argument('-N', '--Ngal', metavar='N', type=int, help='Number of Galaxies', default=200)
    args = ap.parse_args()
    ### -- this was used to find galaxy averages -- ###
    # number = 40
    # specieslist = ['S0', 'Sa', 'Sb', 'Sc', 'SBa', 'SBb', 'SBc', 'cD', 'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
    # for species in specieslist:
    #     galaxies = []
    #     for i in tqdm(range(number)):
    #         galaxies.append(Galaxy(species, (0, 0, 0)))
    #     # galaxies = [Galaxy(species, (0, 0, 0)) for i in range(number)]
    #     masses = [galaxy.galaxymass for galaxy in galaxies]
    #     radii = [galaxy.radius for galaxy in galaxies]
    #     bluef = []; greenf = []; redf = []
    #     for galaxy in galaxies:
    #         for star in galaxy.stars:
    #             bluef.append(star.bandlumin[0])
    #             greenf.append(star.bandlumin[1])
    #             redf.append(star.bandlumin[2])
    #     meanbluef, meangreenf, meanredf = np.mean(bluef), np.mean(greenf), np.mean(redf)
    #     sdbluef, sdgreenf, sdredf = np.std(bluef), np.std(greenf), np.std(redf)
    #     print(f"Galaxy Mass for {number} {species} galaxies: Mean =", np.mean(masses), "with SD =", np.std(masses))
    #     print(f"Galaxy Radius for {number} {species} galaxies: Mean =", np.mean(radii), "with SD =", np.std(radii))
    #     print(f"Galaxy bandlumin for {number} {species} galaxies: Mean =", [meanbluef, meangreenf, meanredf], 
    #           "with SD =", [sdbluef, sdgreenf, sdredf])
    
    ### -- this is the function that you should run! -- ###
    # sim = UniverseSim(1000, mode="Normal")
    # sim.save_data()
    
    # sim = UniverseSim(200, seed=6683)
    sim = UniverseSim(args.Ngal, seed=args.seed)
    sim.save_data(rotcurves=True)
    
    # sim = UniverseSim(800, seed=1000, isotropic=False, homogeneous=False, blackholes=False, darkmatter=True)
    # sim.save_data(proj="Cube", radio=False, rotcurves=True)
    
    # sim = UniverseSim(800, seed=9999, isotropic=True, homogeneous=True, blackholes=True, darkmatter=False)
    # sim.save_data(proj="Cube", radio=False, rotcurves=True)
    
    # for numclust in [400, 800]:
    #     for proj in ["Cube", "DivCube"]:
    #         for pretty in [True, False]:
    #             print(numclust, proj, pretty)
    #             sim = UniverseSim(numclust, seed=3948)
    #             sim.save_data(proj=proj, pretty=pretty, radio=False)
    

    
if __name__ == "__main__":
    main()
