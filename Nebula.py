# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:21:53 2022

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import ListedColormap
import os
import pickle
import MiscTools as misc

class Nebula(object):
    ''' An all-encompassing class for plotting galaxy nebulosity, and planetary nebulae. 
    '''
    def __init__(self, species, position, radius=None, cartesian=False, rotation=None, localgalaxy=False, reduction=True):
        ''' An object to represent (graphically) planetary nebulae or galaxy nebulosity for valid galaxy types according to the
        Galaxy class.
        Parameters
        ----------
        species : str
            The type of nebula (or galaxy type, if wanting galaxy nebulosity). One of {'ring'} or any galaxy type
            valid in the Galaxy class
        position : list or np.array
            Cartesian or spherical coordinates of the center of the nebula. [x, y, z] or [equat, polar, radius] respectively
        radius : float or None
            Radius of the nebula. If it is a galaxy type, must input the radius of the Galaxy object. If it is a typical nebula,
            the size is chosen automatically
        cartesian : bool
            True if the given position coordinates are in cartesian form
        rotation : list or None
            List of values to rotate the nebula in 3D space according to the function MiscTools > CartesianRotation
            If None, the rotation is chosen randomly. If galaxy nebulosity, this value must be the same as the galaxy rotation.
        localgalaxy : bool
            If this nebula corresponds to the nebulosity of the local (host) galaxy of our civilisation
        '''
        self.species = species
        self.radius = radius
        self.local = localgalaxy
        self.reduce = reduction
        self.nebula_params()
        self.rotation = rotation if rotation is not None else np.random.uniform(0, 2*np.pi, 3)
        # self.cmap = self.initColourMap(self.palette)
        self.cartesian = position if cartesian == True else misc.spherical_to_cartesian(position[0], position[1], position[2])
        self.spherical = misc.cartesian_to_spherical(position[0], position[1], position[2]) if cartesian == True else position
        self.points = self.gen_points()
        
    def nebula_params(self):
        ''' Determine some parameters about the nebula based on its type. (radius, colour palette [for plotting])
        This will also create a directory to store custom matplotlib colourmaps (if it doesn't already exist). If the
        desired colourmap doesn't exist, it will call the initColourMap method to create it, otherwise it will load it.
        '''
        if self.species == "ring":
            self.radii = np.array([0.15, 0.12, 0.08]) * 5 if self.radius == None else self.radius
            self.palette = "Hubble"
        elif self.species[0] in ["E", "c"] or self.species == "S0":
            self.radii = self.radius
            self.palette = 'Elliptical'
        elif self.species[0] == "S":
            self.radii = self.radius
            self.palette = 'Spiral'
        
        # now to create or load the colourmap for this nebulous object
        directory = os.path.dirname(os.path.realpath(__file__))    # this is where this .py file is located on the system
        mapdirectory = directory + "/Colourmaps"  # this is the name of the colourmap directory
        if not os.path.exists(mapdirectory): # then we need to create the directory for the colourmap
            os.makedirs(mapdirectory)
        self.paletteDir = mapdirectory + f"/{self.palette}.pickle" # this should be the name of the colour palette file
        if not os.path.isfile(self.paletteDir): # if the colour palette doesn't exist,
            self.cmap = self.initColourMap(self.palette) # create it
        else: # if the colour palette does exist, 
            with open(self.paletteDir, 'rb') as f: 
                self.cmap = pickle.load(f) # load it from file
        
    
    def initColourMap(self, palette):
        ''' Generates a colour map for the species of nebula in question. Returns this colourmap, and also saves it to
        a colour map directory as a pickled object.
        Parameters
        ----------
        palette : str
            The colour palette for plotting and generation of points
        Returns
        -------
        colourmap : list
            List of matplotlib Colourmap objects, with the list length dependent on colour palette.
        '''
        N = 256
        if palette == "Hubble":
            valsR = np.ones((N, 4))
            valsG = np.ones((N, 4))
            valsB = np.ones((N, 4))
            midval = 180
            # start with hubble blue
            valsB[:, 0] = 0
            valsB[:midval, 1] = 178 / 256; valsB[midval:, 1] = np.linspace(178 / 256, 43/256, N - midval)
            valsB[:midval, 2] = 255 / 256; valsB[midval:, 2] = np.linspace(255 / 256, 29/256, N - midval)
            valsB[20:, 3] = np.linspace(0, 0.6, N - 20); valsB[:20, 3] = 0
            HubbleBlue = ListedColormap(valsB)
            # LinearSegmentedColormap('HubbleBlue', valsB)
    
            valsG[:midval, 0] = 200 / 256; valsG[midval:, 0] = np.linspace(200 / 256, 66/256, N - midval)
            valsG[:midval, 1] = 255 / 256; valsG[midval:, 1] = np.linspace(255 / 256, 43/256, N - midval)
            valsG[:, 2] = 0
            valsG[20:, 3] = np.linspace(0, 0.6, N - 20); valsG[:20, 3] = 0
            HubbleGreen = ListedColormap(valsG)
            # LinearSegmentedColormap('HubbleGreen', valsG)
    
            valsR[:midval, 0] = 255 / 256; valsR[midval:, 0] = np.linspace(255 / 256, 66/256, N - midval)
            valsR[:, 1] = 0
            valsR[:, 2] = 0
            valsR[20:, 3] = np.linspace(0, 0.6, N - 20); valsR[:20, 3] = 0
            HubbleRed = ListedColormap(valsR)
            # LinearSegmentedColormap('HubbleRed', valsR)
            
            colourmap = [HubbleRed, HubbleGreen, HubbleBlue]
        elif palette == 'Spiral':
            valsDisk = np.zeros((N, 4))
            valsBulge = np.zeros((N, 4))
            valsArms = np.zeros((N, 4))
            
            midval = 100
            alphamid = 20
            valsDisk[:, 0] = 229 / 256; valsDisk[midval:, 0] = np.linspace(229 / 256, 253/256, N - midval)
            valsDisk[:midval, 1] = 209 / 256; valsDisk[midval:, 1] = np.linspace(209 / 256, 244/256, N - midval)
            valsDisk[:midval, 2] = 199 / 256; valsDisk[midval:, 2] = np.linspace(199 / 256, 239/256, N - midval)
            valsDisk[alphamid:, 3] = np.linspace(0, 0.08, N - alphamid); valsDisk[:alphamid, 3] = 0
            SpiralDisk = ListedColormap(valsDisk)
            
            valsBulge[:, 0] = 229 / 256; valsBulge[midval:, 0] = np.linspace(229 / 256, 253 / 256, N - midval)
            valsBulge[:midval, 1] = 199 / 256; valsBulge[midval:, 1] = np.linspace(199 / 256, 244/256, N - midval)
            valsBulge[:midval, 2] = 189 / 256; valsBulge[midval:, 2] = np.linspace(189 / 256, 239/256, N - midval)
            valsBulge[alphamid:, 3] = np.linspace(0, 0.3, N - alphamid); valsBulge[:alphamid, 3] = 0
            SpiralBulge = ListedColormap(valsBulge)
            
            valsArms[:, 0] = 183 / 256
            valsArms[:, 1] = 185 / 256
            valsArms[:, 2] = 240 / 256
            valsArms[alphamid:, 3] = np.linspace(0, 0.8, N - alphamid); valsArms[:alphamid, 3] = 0
            SpiralArms = ListedColormap(valsArms)
            
            if self.species[1] == "B":
                valsBar = np.zeros((N, 4))
                valsBar[:, 0] = 229 / 256
                valsBar[:, 1] = 199 / 256
                valsBar[:, 2] = 189 / 256
                valsBar[alphamid:, 3] = np.linspace(0, 0.6, N - alphamid); valsBar[:alphamid, 3] = 0
                SpiralBar = ListedColormap(valsBar)
                colourmap = [SpiralDisk, SpiralBulge, SpiralArms, SpiralBar]
            else:
                colourmap = [SpiralDisk, SpiralBulge, SpiralArms]
        elif palette == 'Elliptical':
            valsBulge = np.zeros((N, 4))
            
            midval = 100
            alphamid = 20
        
            valsBulge[:, 0] = 229 / 256; valsBulge[midval:, 0] = np.linspace(229 / 256, 253 / 256, N - midval)
            valsBulge[:midval, 1] = 199 / 256; valsBulge[midval:, 1] = np.linspace(199 / 256, 244/256, N - midval)
            valsBulge[:midval, 2] = 189 / 256; valsBulge[midval:, 2] = np.linspace(189 / 256, 239/256, N - midval)
            valsBulge[alphamid:, 3] = np.linspace(0, 0.3, N - alphamid); valsBulge[:alphamid, 3] = 0
            SpiralBulge = ListedColormap(valsBulge)
            colourmap = [SpiralBulge]
            
        with open(self.paletteDir, 'wb') as f:
            pickle.dump(colourmap, f)
            
        return colourmap
    
    def gen_points(self):
        ''' Basic function to generate correctly distributed points based on the nebula type. There is a distinct 
        set of points for each colourmap in the colour palette. 
        Returns
        -------
        points : list
            List of [x, y, z] coordinates for each object in the colourmap list. E.g. for a colourmap with two entries, 
            points = [[x, y, z], [x, y, z]]
        '''
        if self.species == 'ring':
            points = self.gen_ring_nebula()
        elif self.species[0] in ["E", "c"] or self.species == "S0":
            points = self.gen_elliptical_nebulosity()
        elif self.species[0] == "S":   # dealing with spiral galaxy
            points = self.gen_spiral_nebulosity()
        
        return points
    
    def gen_ring_nebula(self):
        '''
        '''
        pops = [150000, 120000, 100000]
        lowers = [0, 0.1, 0.2]
        coords = []
        for i, colour in enumerate(self.cmap):
            pop = pops[i]
            radius = self.radii[i] * (np.random.uniform(lowers[i], 1, pop))**(1/3)
            # the following distributes points more or less evenly about a sphere centered at the origin
            theta = np.random.uniform(0, 2*np.pi, pop)
            vert = 0.6 if i < 2 else 1
            phi = np.arccos(np.random.uniform(-vert, vert, pop))
            centerx = radius * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.1, pop))
            centery = radius * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.1, pop))
            zmult = 1 if i < 2 else 1.5
            centerz = zmult * radius * (np.cos(phi) * np.random.normal(1, 0.3, pop))
            if i == 0:
                outPop = int(0.75 * pop); outRad = 1.5 * self.radii[i]
                outPhi = np.arccos(np.random.uniform(-1, 1, outPop))
                outTheta = np.random.uniform(0, 2*np.pi, outPop)
                centerx = np.append(centerx, outRad * (np.cos(outTheta) * np.sin(outPhi) * np.random.normal(1, 0.1, outPop)))
                centery = np.append(centery, outRad * (np.sin(outTheta) * np.sin(outPhi) * np.random.normal(1, 0.1, outPop)))
                centerz = np.append(centerz, outRad * (np.cos(outPhi) * np.random.normal(1, 0.3, outPop)))
            
            points = np.array([centerx, centery, centerz])
            points = np.dot(misc.cartesian_rotation(self.rotation[0], 'x'), points)
            points = np.dot(misc.cartesian_rotation(self.rotation[1], 'y'), points)
            points = np.dot(misc.cartesian_rotation(self.rotation[2], 'z'), points)
            
            points[0] += self.cartesian[0]
            points[1] += self.cartesian[1]
            points[2] += self.cartesian[2]
            coords.append(points)
        return coords
    
    def gen_spiral_nebulosity(self):
        '''
        '''
        coords = []
        for i, colour in enumerate(self.cmap):
            
            if self.reduce:
                pop = int(3e5 * np.exp(-self.spherical[2] / 10**3.7))
            else:
                pop = int(3e5)
            
            if i == 0:      # we're dealing with the disk component
                theta = np.random.uniform(0, 2*np.pi, pop)
                diskdists = 1.05 * self.radius * np.random.uniform(0, 1, pop)**(1/2)
                x = np.cos(theta) * diskdists * np.random.normal(1, 0.2, pop)
                y = np.sin(theta) * diskdists * np.random.normal(1, 0.2, pop)
                z = np.zeros(pop) + 0.02 * self.radius * np.random.randn(pop)
            elif i == 1:       # we're dealing with the bulge
                bulgepop = pop
                
                cosphi = np.random.uniform(-1, 1, bulgepop)
                phi = np.arccos(cosphi)
                theta = np.random.uniform(0, 2*np.pi, bulgepop)
                
                bulgeradius = 0.3 * self.radius
                bulgeR = bulgeradius * np.random.uniform(0, 1, bulgepop)**(1/2.5)    #bulgedists was meant to be RVs between 0 and 1, but the mult makes up for it
                x = bulgeR * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.2, bulgepop))
                y = bulgeR * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.2, bulgepop))
                z = 0.7 * (bulgeR * np.cos(phi) + np.random.normal(0, 0.2, bulgepop)) #* 0.7**distanceflat
                
            elif i in [2, 3]:       # we're dealing with spiral arms
                popindex = {"S0": 1000, "Sa": 3000, "Sb": 4000, "Sc": 10000, "SBa": 10000, "SBb": 10000, "SBc": 10000}
                if self.reduce:
                    pop = int(popindex[self.species] * np.exp(-self.spherical[2] / 10**4))
                else:
                    pop = popindex[self.species]
                speciesindex = {"S0":0, "Sa":1, "Sb":2, "Sc":3, "SBa":4, "SBb": 5, "SBc":6}
                wrap = [[None, None], [0.9, 4 * np.pi], [0.7, 2 * np.pi], [0.2, 1 * np.pi], 
                        [np.pi / 2.1, 3 * np.pi], [np.pi / 2.1, 2.1 * np.pi], [np.pi / 2.1, 1.4 * np.pi]]
                
                if i == 2:
                    #now to actually grab the parameters for the galaxy type in question:
                    SpiralRadiiDiv = [None, 15, 7, 2.1, 3.7, 3, 2.3] 
                    mult, spiralwrap = [param[speciesindex[self.species]] for param in [SpiralRadiiDiv, wrap]]
                    upper, lower = spiralwrap
                    if self.species in ["Sa", "Sb"]:
                        spiralangle = np.linspace(lower**2, upper**1.6, pop)
                        spiralangle = np.sqrt(spiralangle)
                    elif speciesindex[self.species] > 5:
                        spiralangle = np.geomspace(lower, upper, pop)
                    else:
                        spiralangle = np.linspace(lower, upper, pop)
                    reflect = np.random.choice([-1, 1], pop)
                    # power = 1/2 if self.species[:2] == "SB" else 1
                    spiralpow = np.sqrt(spiralangle) if self.species[:2] == "SB" else spiralangle
                    [lag, scatter, scatter2, zscatter] = [0, 0.1, 0.1, 0.03]
                    x = (self.radius / mult) * (spiralpow * np.cos(spiralangle + lag)  * np.random.normal(1, scatter, pop) * reflect + np.random.normal(0, scatter2, pop))
                    y = (self.radius / mult) * (spiralpow * np.sin(spiralangle + lag) * np.random.normal(1, scatter, pop) * - reflect + np.random.normal(0, scatter2, pop))
                    z = np.random.normal(0, zscatter * self.radius, pop)
                elif i == 3:    # we're dealing with spiral bar
                    barradii = [0, 0, 0, 0, 0.3, 0.4, 0.5]
                    barradius = barradii[speciesindex[self.species]] * self.radius
                    x = np.random.normal(0, 0.3 * barradius, pop)
                    y = barradius * (np.linspace(0, 1.1, pop) * np.random.choice([-1, 1], pop) + np.random.normal(0, 0.3, pop))
                    z = np.random.normal(0, 0.1 * barradius, pop)
            
            points = np.array([x, y, z])
            points = np.dot(misc.cartesian_rotation(self.rotation[0], 'x'), points)
            points = np.dot(misc.cartesian_rotation(self.rotation[1], 'y'), points)
            points = np.dot(misc.cartesian_rotation(self.rotation[2], 'z'), points)
            
            points[0] += self.cartesian[0]
            points[1] += self.cartesian[1]
            points[2] += self.cartesian[2]
            coords.append(points)
        return coords
    
    def gen_elliptical_nebulosity(self):
        '''
        '''
        if self.reduce:
            pop = int(3e5 * np.exp(-self.spherical[2] / 10**3.7))
        else:
            pop = int(3e5)
            
        ellipsoid_mult = (1 - float(self.species[1]) / 10) if self.species[0]=='E' else 1
        
        theta = np.random.uniform(0, 2*np.pi, pop)
        phi = np.random.uniform(-1, 1, pop)
        phi = np.arccos(phi)
        
        if self.species == "S0":
            diskdists = self.radius * np.random.uniform(0, 1, pop)**(1/2)
            x = np.cos(theta) * diskdists * np.random.normal(1, 0.2, pop)
            y = np.sin(theta) * diskdists * np.random.normal(1, 0.2, pop)
            z = np.zeros(pop) + 0.08 * self.radius * np.random.randn(pop)
            
            bulgepop = int(pop / 15)
            
            cosphi = np.random.uniform(-1, 1, bulgepop)
            phi = np.arccos(cosphi)
            theta = np.random.uniform(0, 2*np.pi, bulgepop)
            
            bulgeradius = 0.3 * self.radius
            bulgeR = bulgeradius * np.random.uniform(0, 1, bulgepop)**(1/2.5)    #bulgedists was meant to be RVs between 0 and 1, but the mult makes up for it
            bulgex = bulgeR * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.2, bulgepop))
            bulgey = bulgeR * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.2, bulgepop))
            #distanceflat = (1 / self.radius) * np.sqrt(np.square(bulgex) + np.square(bulgey))     #this makes the z lower for stars further from the center
            bulgez = 0.5 * (bulgeR * np.cos(phi) + np.random.normal(0, 0.2, bulgepop)) #* 0.7**distanceflat
            x = np.append(x, bulgex); y = np.append(y, bulgey); z = np.append(z, bulgez)
            
        else:
            spheredists = np.random.exponential(0.4, pop)
            sphereR = 2 * self.radius * np.sqrt(spheredists)
            
            sphereR = 1.5 * self.radius * np.random.uniform(0, 1, pop)**(1/2)
            
            x = sphereR * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.2, pop))
            y = sphereR * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.2, pop))
            distanceflat = (1 / self.radius) * np.sqrt(np.square(x) + np.square(y))
            z = (sphereR * (np.cos(phi) + np.random.normal(0, 0.1, pop))) * ellipsoid_mult**distanceflat
    
            
        points = np.array([x, y, z])
        points = np.dot(misc.cartesian_rotation(self.rotation[0], 'x'), points)
        points = np.dot(misc.cartesian_rotation(self.rotation[1], 'y'), points)
        points = np.dot(misc.cartesian_rotation(self.rotation[2], 'z'), points)
        
        points[0] += self.cartesian[0]
        points[1] += self.cartesian[1]
        points[2] += self.cartesian[2]
        
        coords = [points]
        return coords
        
        
    def plot_nebula(self, figAxes=None, style='colormesh', method='AllSky'):
        '''
        Parameters
        ----------
        figAxes : list (or None)
            List in the form of [fig, ax] (if AllSky projection), or [[fig1, ax1], [fig2, ax2],...,[fig6, ax6]] if cubemapped.
        style : str
            One of {'colormesh', 'imshow', 'hexbin'}
        method : str
            One of {"AllSky", "Cube"}
        '''
        if self.palette == 'Spiral':
            bins = 200 if not self.reduce else int(200 * np.exp(-self.spherical[2] / 10**4.4))
            # grid = 120 if not self.reduce else int(200 * np.exp(-self.spherical[2] / 10**4.4))
        else:
            bins = 150 if not self.reduce else int(200 * np.exp(-self.spherical[2] / 10**4.4))
            # grid = 50 if not self.reduce else int(200 * np.exp(-self.spherical[2] / 10**4.4))
        if figAxes == None:
            figAxes = misc.gen_figAxes(method=method)
        else:
            if method=="AllSky":
                fig, ax = figAxes
        for i, colour in enumerate(self.cmap):
            vmax = None
                
            if self.palette == 'Spiral':
                smooth = 2.5
            else:
                smooth = 1.5
            x, y, z = self.points[i]
            if method == "Cube":
                origBins = bins
                if self.local:
                    smooth = 5
                uc, vc, index = misc.cubemap(x, y, z)
                maxDens = 0
                EXYD = []
                for i in range(6):
                    X, Y = uc[index == i], vc[index == i] # get all coords of points on this cube face
                    if ((len(X) <= 1e3 or len(Y) <= 1e3) and i<2) or ((len(X) <= 1e2 or len(Y) <= 1e2) and i>=2): 
                        # if len(points) <= 1000 for a bulge/elliptical galaxy, or <=100 for spiral arms/bulge
                        EXYD.append([])
                        continue # this stops extremely patchy sections of nebulosity
                    if not self.local:
                        # now, we need to make cut-off nebulae smoother, by reducing the number of bins proportionally to how many
                        # points have *not* been cut off
                        bins = origBins
                        bins *= max(np.cbrt(len(X) / len(x)), np.cbrt(len(Y) / len(y)))
                        bins = int(bins)
                    extent = [[min(X) - 1, max(X) + 1], [min(Y) - 1, max(Y) + 1]]
                    density, Xedges, Yedges = np.histogram2d(X, Y, bins=[2 * bins, bins], range=extent)
                    Xbins = Xedges[:-1] + (Xedges[1] - Xedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
                    Ybins = Yedges[:-1] + (Yedges[1] - Yedges[0]) / 2
                    
                    density = density.T      # take the transpose of the density matrix
                    density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
                    Xbins = scipy.ndimage.zoom(Xbins, 2)
                    Ybins = scipy.ndimage.zoom(Ybins, 2)
                    density = scipy.ndimage.gaussian_filter(density, sigma=smooth)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
                    maxDens = np.amax(density) if np.amax(density) > maxDens else maxDens
                    EXYD.append([extent, Xbins, Ybins, density])
                # now let's actually plot the nebulosity
                for i in range(6):
                    if EXYD[i] == []:
                        continue
                    extent, Xbins, Ybins, density = EXYD[i]
                    if style == 'colormesh':
                        figAxes[i][1].grid(False)
                        figAxes[i][1].pcolormesh(Xbins, Ybins, density, cmap=colour, vmax=maxDens, shading='auto', rasterized=True, antialiased=True)
                        figAxes[i][1].grid(True)
                    elif style == 'imshow':
                        extent = [min(X), max(X), min(Y), max(Y)]
                        figAxes[i][1].imshow(density, extent=extent, cmap=colour, interpolation='none')
            else:
                equat, polar, distance = misc.cartesian_to_spherical(x, y, z)
                    
                extent = [[min(equat) - 3, max(equat) + 3], [min(polar) - 3, max(polar) + 3]]   # this is so that the edge of the contours aren't cut off
                density, equatedges, polaredges = np.histogram2d(equat, polar, bins=[2 * bins, bins], range=extent)
                equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
                polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2
                
                density = density.T      # take the transpose of the density matrix
                density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
                equatbins = scipy.ndimage.zoom(equatbins, 2)
                polarbins = scipy.ndimage.zoom(polarbins, 2)
                # if self.palette == 'Spiral:'
                density = scipy.ndimage.gaussian_filter(density, sigma=smooth)  # this smooths the area density even moreso (not necessary, but keeping for posterity)
                
                if style == 'colormesh':
                    # import matplotlib.colors as colors
                    # ax.pcolormesh(equatbins, polarbins, density, cmap=colour, shading='auto', rasterized=True, 
                    #               norm=colors.PowerNorm(gamma=0.8))
                    ax.pcolormesh(equatbins, polarbins, density, cmap=colour, vmax=vmax, shading='auto', rasterized=True, antialiased=True)
                elif style == 'imshow':
                    extent = [min(equat), max(equat), min(polar), max(polar)]
                    ax.imshow(density, extent=extent, cmap=colour, interpolation='none')


# img = plt.imread("Datasets/Sim Data (Clusters; 1000, Seed; 588)/Universe Image.png")


# ax.imshow(img, extent=[0, 360, 0, 180])

def main():
    # ringNeb = Nebula('ring', [45, 90, 10])
    # ringNeb.plot_nebula(style='hexbin')
    from Galaxy import Galaxy
    # position = [45, 90, 1000]
    # species = 'E0'
    # galax = Galaxy(species, position)
    # fig, ax = plt.subplots()
    
    # spiralNeb = Nebula(species, position, galax.radius, rotation=galax.rotation)
    # spiralNeb.plot_nebula(style='colormesh', ax=ax)
    # galax.plot_2d(fig, ax)
    # ax.set_xlim(40, 50)
    # ax.set_ylim(95, 85)
    
    species = 'SBa'
    position = [225, 90, 600]
    # position = [180, 90, 40]
    galax = Galaxy(species, position, rotate=True)
    # fig, ax = plt.subplots()
    
    spiralNeb = Nebula(species, position, galax.radius, rotation=galax.rotation, localgalaxy=False)
    spiralNeb.plot_nebula(style='colormesh', method="Cube", localgalaxy=False)
    
    # # ringNeb = Nebula('ring', [150, 85, 10])
    # # ringNeb.plot_nebula(ax=ax)
    
    # galax.plot_2d(fig, ax)
    # fig.set_size_inches(18, 9, forward=True)
    # fig.savefig('galax.png', dpi=1500, bbox_inches='tight', pad_inches = 0.01)
    
    
    
    
if __name__ == "__main__":
    main()




