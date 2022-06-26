# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:52:26 2022

@author: ryanw

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
import scipy.optimize as opt             # this is to fit two axes in the HR diagram
import scipy.ndimage                     # this is to smooth out the BH radio lobes
from multiprocessing import Pool
# import colour as col

class BlackHole(object):
    def __init__(self, galaxymass, galaxytype, galaxyradius, luminosity):
        '''
        Parameters
        ----------
        galaxymass : float
            baryonic mass of the galaxy in solar masses
        luminosity : float
            the fraction of the eddington luminosity that the black hole should be
        '''
        self.mass = self.initialise_mass(galaxymass)
        self.luminosity = luminosity * self.eddington_luminosity(self.mass)
        self.galaxyradius = galaxyradius
        if galaxytype[0] == "S":
            self.BHradio = self.BH_emission(FR=1)
        else:
            self.BHradio = self.BH_emission(FR=2)
    
    def initialise_mass(self, galaxymass):
        return (galaxymass * 3 * 10**-2) * np.random.normal(1, 0.1)
    
    def eddington_luminosity(self, mass):
        ''' Eddington luminosity for an accreting black hole. 
        '''
        return 3 * 10**4 * (mass)
    
    def get_BH_mass(self):
        return self.mass
    def get_BH_lumin(self):
        return self.luminosity
    def get_BH_colour(self):
        ''' Quasar RGB colour. 
        '''
        return np.array([73, 214, 255]) / 255
    def get_BH_radio(self):
        return self.BHradio
    
    def get_BH_scale(self):
        '''A basic logarithmic scale to determine how large stars should be in pictures given their luminosity.
        Returns
        -------
        scale : float
            the number to be input into the 'scale' kwarg in a matplotlib figure. 
        '''
        scale = 3 * np.log(2 * self.luminosity + 1)
        scale = 2 if scale > 2 else scale
        return scale
    
    def BH_emission(self, FR=2):
        ''' Produce mock-radio emission lobes from the SMBH in the center of the galaxy according to its Fanaroff-Riley type, FR. 
        Inspiration for shaping came from  A.H. Bridle et al. 1994, Deep VLA imaging of twelve extended 3CR quasars, AJ 108, 766,
        available at : https://articles.adsabs.harvard.edu/pdf/1994AJ....108..766B
        This function emulates emission lobes by randomly scattering dots in 3D space according to rules and specified positions:
            1. A central core, with 'centerpop' dots and radius 0.1pc
            2. Jets (non-symmetric if FR-2), extending out of the galactic plane
            3. Lobes, which expand out from the ends of the jet. 
        This function purely creates the dots. It is the job of other functions (Galaxy.galaxy_radio) to plot the area density ("intensity")
        Parameters
        ----------
        FR : int
            {1, 2} depending on which Fanaroff-Riley emission type to simulate. 
        '''
        if FR == 1:
            centerpop = 500
            centerradius = 0.1
            # the following distributes points more or less evenly about a sphere centered at the origin
            theta = np.random.uniform(0, 2*np.pi, centerpop)
            phi = np.random.uniform(-1, 1, centerpop)
            phi = np.arccos(phi)
            centerx = centerradius * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centery = centerradius * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centerz = centerradius * (np.cos(phi) * np.random.normal(1, 0.3, centerpop))
            
            jetpop = 1000
            jetradius = 0.5 * self.galaxyradius
            jetz = jetradius * (np.geomspace(0.01, 1.6, jetpop) * np.random.normal(1, 0.01, jetpop))    
            jetx = np.random.normal(0, 0.2 * jetz, jetpop)
            jety = np.random.normal(0, 0.2 * jetz, jetpop)
            
            jetreflect = np.random.uniform(0, 1, jetpop)
            for i, val in enumerate(jetreflect):
                if val > 0.5:         # half of the points are reflected (symmetrical)
                    jetz[i] *= -1; jetx[i] *= -1; jety[i] *= -1
                    
            lobepop = 4000
            loberadius = jetradius * 2
            mult = 5    # arbitrary divisor to compact the lobes towards the jet a little bit
            lobeangle = np.geomspace(0.5 * np.pi, 1.5 * np.pi, lobepop)
            lobex = loberadius / mult * (lobeangle * np.cos(1.2 * lobeangle) * np.random.normal(1, 0.1, lobepop) + np.random.normal(0, 0.3 * lobeangle, lobepop))
            lobey = loberadius / mult * (lobeangle * np.sin(1.2 * lobeangle) * np.random.normal(1, 0.1, lobepop) + np.random.normal(0, 0.3 * lobeangle, lobepop)) - jetradius/2
            lobez = 0.5 * jetradius + (loberadius / mult * (lobeangle * np.sin(0.7 * lobeangle) * np.random.normal(1, 0.2, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop)))
            
            lobereflect = np.random.uniform(0, 1, lobepop)
            for i, val in enumerate(lobereflect):
                if val > 0.5:       # half of the points are reflected
                    lobez[i] *= -1; lobex[i] *= -1; lobey[i] *= -1  # reflect the coordinates about their respective axes. 
        # the next section is functionally identical - i couldn't really be bothered reducing the linecount here
        else:
            centerpop = 100
            centerradius = 0.1
            theta = np.random.uniform(0, 2*np.pi, centerpop)
            phi = np.random.uniform(-1, 1, centerpop)
            phi = np.arccos(phi)
            centerx = centerradius * (np.cos(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centery = centerradius * (np.sin(theta) * np.sin(phi) * np.random.normal(1, 0.1, centerpop))
            centerz = centerradius * (np.cos(phi) * np.random.normal(1, 0.3, centerpop))
            
            jetpop = 1000
            jetradius = 2.5 * self.galaxyradius
            jetz = jetradius * (np.linspace(0.01, 1, jetpop) * np.random.normal(1, 0.01, jetpop))
            jetx = np.random.normal(0, 0.01 * jetz, jetpop)
            jety = np.random.normal(0, 0.01 * jetz, jetpop)
            
            jetreflect = np.random.uniform(0, 1, jetpop)
            for i, val in enumerate(jetreflect):
                if val > 0.9:       # only about 10% of the points are reflected
                    jetz[i] *= -1; jetx[i] *= -1; jety[i] *= -1
            
            lobepop = 4000
            loberadius = jetradius * 2/3
            mult = 5    # arbitrary divisor to compact the lobes towards the jet a little bit
            lobeangle = np.geomspace(0.5 * np.pi, 1 * np.pi, lobepop)
            lobex = loberadius / mult * (lobeangle * np.cos(lobeangle) * np.random.normal(1, 0.1, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop))# * reflect 
            lobey = loberadius / mult * (np.random.normal(0, 0.1, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop))# * - reflect + np.random.normal(0, scatter2, lobepop))
            lobez = 0.7 * jetradius + (loberadius / mult * (lobeangle * np.sin(lobeangle) * np.random.normal(1, 0.2, lobepop) + np.random.normal(0, 0.2 * lobeangle, lobepop)))
            
            lobereflect = np.random.uniform(0, 1, lobepop)
            for i, val in enumerate(lobereflect):
                if val > 0.6:
                    lobez[i] *= -1; lobex[i] *= -1; lobey[i] *= -1
            
        x = np.concatenate((centerx, jetx, lobex), axis=0)
        y = np.concatenate((centery, jety, lobey), axis=0)
        z = np.concatenate((centerz, jetz, lobez), axis=0)
        radius = centerradius + jetradius + loberadius
        return x, y, z, radius
        
        

class Star(object):
    def __init__(self, location, species="MS"):
        self.colourdata = pd.read_csv("blackbodycolours.txt", delimiter=' ')
        if species == "MS":
            self.mass = abs(self.MS_masses(location))
            self.luminosity = abs(self.MS_lumin(self.mass))
            self.radius = self.MS_radius(self.mass)
            self.temperature = self.MS_temperature(self.luminosity, self.radius)
        elif species == "WDwarf":
            self.mass = abs(self.WD_masses())
            self.radius = self.WD_radii(self.mass)
            self.temperature = self.WD_temp(self.mass)
            self.luminosity = abs(self.WD_lumin(self.temperature, self.radius, self.mass))
        elif species == "Giant":
            self.temperature = self.giant_temp()
            self.luminosity = abs(self.giant_lumin(self.temperature))
            self.mass = abs(self.giant_mass())
            self.radius = self.stefboltz_radius(self.luminosity, self.temperature)
        elif species == "SupGiant":
            self.temperature = self.SGB_temp()
            self.luminosity = abs(self.SGB_lumin(self.temperature))
            self.mass = abs(self.SGB_mass())
            self.radius = self.stefboltz_radius(self.luminosity, self.temperature)
        self.bandlumin = self.generate_BandLumin(self.temperature, self.radius)
            
    def MS_masses(self, species):
        '''Masses for stars on the main sequence.
        Parameters
        ----------
        species : str
            which part of the galaxy the star is in.
        '''
        if species in ("youngspiral", "ys"):
            a, b = 2, 2.5
        elif species in ("oldspiral", "os"):
            a, b = 1, 2
        elif species == "disk":
            a, b = 1, 0.5
        elif species == "bulge":
            a, b = 1, 1
        return np.random.gamma(a, b) + 0.08
    
    def MS_lumin(self, mass):
        '''Piecewise relationships taken from:
            Wikipedia : https://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
            Evolution of Stars and Stellar Populations (book) : https://cfas.org/data/uploads/astronomy-ebooks/evolution_of_stars_and_stellar_populations.pdf
        Parameters
        ----------
        mass : float
            solar masses of the star
        Returns
        -------
        lumin : float
            solar luminosities of the star
        '''
        if 0.08 <= mass <= 0.43:
            lumin = 0.23 * mass**2.3
        elif 0.43 < mass <= 2:
            lumin = mass**4
        elif 2 < mass <= 55:
            lumin = 1.4 * mass**3.5
        elif 55 < mass:
            lumin = 3.2 * 10**4 * mass
        else: ValueError("something is wrong")
        lumin += np.random.normal(0, 0.005 * mass)
        lumin = 0.23 * 0.04**2.3 if lumin < 0.23 * 0.04**2.3 else lumin
        return lumin

    def MS_radius(self, mass):
        '''Piecewise relationships taken from:
            https://articles.adsabs.harvard.edu/pdf/1991Ap%26SS.181..313D
            (with some tweaking)
        
        ZAMS = zero-age main sequence (new star)
        TAMS = terminal age main sequence (about to transition off of MS)
        '''
        if mass < 1.66:
            TAMS = 2 * mass**0.75
            ZAMS = 0.89 * mass**0.89
        elif mass >= 1.66:
            TAMS = 1.61 * mass**0.83
            ZAMS = 1.31 * mass**0.57
        ave = (TAMS + ZAMS) / 2
        radius = np.random.normal(ave, 0.12 * mass)
        return radius
        
    def MS_temperature(self, lumin, radius):
        '''Main sequence temperature using stefan-boltzman equation. 
        '''
        sigma = 5.67037 * 10**-8
        R = 696340000 * radius
        L = 3.828 * 10**26 * lumin
        temp = (L / (sigma * 4 * np.pi * R**2 ))**(1/4)
        temp += np.random.normal(0, 0.1 * temp)
        temp = 40000 if temp > 40000 else temp
        return temp
    
    def WD_masses(self):
        '''Masses for white dwarf stars.
        Strong peak at M ~ 0.65, and must be in the range [0.17, 1.33]
        Distribution retrieved from Fig 1 from:
            https://www.lume.ufrgs.br/bitstream/handle/10183/90266/000586456.pdf?sequence=1
        Four different normal distributions make up the total distribution, with mass fractions 7%, 69%, 23% and 1% respectively. 
        
        Parameters
        ----------
        n : int
            the number of white dwarfs to simulate
        
        Returns
        -------
        masses : np.array or float64
            The mass of the white dwarf stars in units Solar Masses
        '''
        prob = np.random.uniform(0, 1)
        if prob <= 0.07:
            mass = np.random.normal(0.38, 0.05)
        elif prob <= 0.69+0.07:
            mass = np.random.normal(0.578, 0.05)
        elif prob <= 0.23 + 0.69 + 0.07:
            mass = np.random.normal(0.658, 0.2)
        else:
            mass = np.random.normal(1.081, 0.2)
        mass = 1.33 if mass > 1.33 else mass
        mass = 0.17 if mass < 0.17 else mass
        return mass
    
    def WD_radii(self, mass):
        '''Radius ~ M^(-1/3) from wikipedia :
            https://en.wikipedia.org/wiki/White_dwarf
        '''
        radii = 6 * 10**-3 * mass**(-1/3) * np.random.normal(1, 0.01)
        return radii

    def WD_temp(self, mass):
        '''I don't remember where this came from -- i think I may have curve-fit wikipedia data on a log scale. 
        '''
        return 10**(0.7 * np.log10(mass) + 4.4) * np.random.normal(1, 0.05)

    def WD_lumin(self, temps, radii, masses):
        ''' Uses the Stefan-Boltzmann equation with some multiplier to calculate lumin of white dwarf. 
        Parameters
        ----------
        masses : float or np.array
            masses of the white dwarf stars in units of solar masses
            
        Returns
        -------
        luminosity : float
            Luminosity of the white dwarf in solar luminosities
        '''
        sigma = 5.67037 * 10**-8
        R = 696340000 * radii
        solLum = 3.828 * 10**26
        mult =  1 / solLum
        return 4 * np.pi * R**2 * sigma * temps**4 * mult * np.random.normal(1, 0.1)

    def SGB_temp(self):
        '''Beta temperature distribution, weighted to be just higher in temperature than the midpoint. 
        Returns
        -------
        temp : float
            temp in Kelvin. 
        '''
        a, b = 2.5, 2
        mintemp, maxtemp = 2000, 2.2 * 10**4
        temp = (np.random.beta(a, b) * maxtemp) + mintemp
        return temp * np.random.normal(1, 0.2)

    def SGB_lumin(self, temp):
        ''' Modelled to be a inverted parabola, given start and end points at the low and high temperature extrema,
        and to be around 10^4.5 solar luminosities. 
        Returns
        -------
        luminosity : float
            solar luminosities of the supergiant star. 
        '''
        a, b, c = -0.0076, 206.516, -350972
        lumin = a * temp**2 + b * temp + c
        return lumin * np.random.normal(1, 0.3)
    
    def SGB_mass(self):
        ''' Normally distributed mass centered at 20 solar masses with SD of 8. 
        '''
        return np.random.normal(10, 8) + 10

    def giant_temp(self):
        ''' Gamma distribution of star temperatures, weighted to be lower (~4000). 
        '''
        a, b = 4, 1
        temp = 1000 * np.random.gamma(a, b) + 2000
        return temp * np.random.normal(1, 0.1)

    def giant_lumin(self, temp):
        ''' Not entirely sure what's going on here. Returns luminosity in solar luminosities. 
        '''
        a, b, c = 10**-7 * 2.857, -0.00343, 10.71
        add = 10**4.3 * np.exp(a*(temp-4000)**2 + b*temp + c) #what the heck is this (it adds extra luminosity to low temp stars)
        return (temp + add) * abs(np.random.normal(1, 0.8)) * 0.02
    
    def giant_mass(self):
        ''' Normally distributed mass centered at 8 solar masses with SD of 3. 
        '''
        return np.random.normal(4, 3) + 4 
    
    def stefboltz_radius(self, lumin, temp):
        ''' Stefan-Boltzman equation to calculate radius. 
        Parameters
        ----------
        lumin, temp : float
            the solar luminosities and temp (K) of the star
        Returns
        -------
        radius : float
            solar radii of the star. 
        '''
        sigma = 5.67037 * 10**-8
        lumin = 3.828 * 10**26 * lumin
        radius = np.sqrt(lumin / (4 * np.pi * sigma * temp**4))
        return radius / 696340000

    def get_star_colour(self):
        ''' Takes the RGB colour values from a blackbody of specified temperature, with values stored in the file:
            "blackbodycolours.txt"
        Blackbody RGB values kindly supplied by:
            Mitchell Charity <mcharity@lcs.mit.edu>
            http://www.vendian.org/mncharity/dir3/blackbody/
            Version 2001-Jun-22
        
        Alternate version (commented out) using colour-science package:
            Approximations were retrieved from https://en.wikipedia.org/wiki/Planckian_locus
        '''
        #i tried to use an algorithm here but i ran into issues with the colour-science (or is it colour?) package
        # temp = self.temperature
        # if 1667 <= temp <= 4000:
        #     x = -0.2661239 * (10**9 / temp**3) - 0.2343589 * (10**6 / temp**2) + 0.8776956 * (10**3 / temp) + 0.179910
        #     if temp <= 2222:
        #         y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
        #     else:
        #         y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
        # elif 4000 <= temp <= 25000:
        #     x = -3.0258469 * (10**9 / temp**3) + 2.1070379 * (10**6 / temp**2) + 0.2226347 * (10**3 / temp) + 0.24039
        #     y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
        # xy = [x, y]
        # XYZ = col.xy_to_XYZ(xy)
        # rgb = col.XYZ_to_RGB(XYZ)
        # return rgb
        temperature = self.temperature
        temperature = 40000 if temperature > 40000 else temperature
        temperature = 1000 if temperature < 1000 else temperature
        temp = round(temperature / 100) * 100
        r, g, b = self.colourdata.loc[self.colourdata['Temperature'] == temp].iloc[0, 9:12]
        rgb = np.array([r, g, b]) / 255
        return rgb
    
    def get_star_scale(self):
        '''A basic logarithmic scale to determine how large stars should be in pictures given their luminosity.
        Returns
        -------
        scale : float
            the number to be input into the 'scale' kwarg in a matplotlib figure. 
        '''
        scale = 3 * np.log(2 * self.luminosity + 1)
        scale = 2 if scale > 2 else scale
        return scale
    
    def generate_BandLumin(self, temp, radius):
        ''' Equations retrieved from: http://burro.cwru.edu/academics/Astr221/Light/blackbody.html
        The returned band luminosities are at wavelengths: B = 440nm, G = 500nm, R = 700nm
        Each luminosity value has an uncertainty of +/- 1.5% about a true blackbody. 
        Parameters
        ----------
        temp : float
        radius : float
            Radius of the star in units of solar radii. 
        Returns
        -------
        band luminosities : np.array
            [B, G, R] band luminosities in units of J/nm/s <=> W/nm <=> 10^-9 W/m
            (the planck function has units of J/m^2/nm/s <=> W/m^2/nm <=> 10^-9 W/m^3 )
        '''
        c, h, k = 299792458, 6.626 * 10**-34, 1.38 * 10**-23
        blue, green, red = 440 * 10**-9, 500 * 10**-9, 700 * 10**-9
        planck = lambda x: ((2 * h * c**2) / x**5) * (1 / (np.exp(h * c / (x * k * temp)) - 1)) * 10**-9
        bandLum = lambda x: 4 * np.pi**2 * (696540000 * radius)**2 * planck(x)
        return np.array([bandLum(blue), bandLum(green), bandLum(red)]) * np.random.uniform(0.99, 1.01, 3)
    
    def generate_blackbodycurve(self, markers=True, visible=False):
        ''' Produce a graph of this stars' blackbody curve. 
        Parameters
        ----------
        markers : bool
            whether or not to put the [B, G, R] band luminosity markers on the graph
        visible : bool
            whether or not to plot the visible spectrum overlaid onto the curve
        '''
        temp = self.temperature
        radius = self.radius
        c, h, k = 299792458, 6.626 * 10**-34, 1.38 * 10**-23
        x = np.linspace(1 * 10**-9, 10**-6, 1000)   # the domain for the graph. going from ~0 -> 1000nm
        planck = lambda x: ((2 * h * c**2) / x**5) * (1 / (np.exp(h * c / (x * k * temp)) - 1)) * 10**-9
        bandLum = lambda x: 4 * np.pi**2 * (696540000 * radius)**2 * planck(x)
        lumins = bandLum(x)     # generate the blackbody curve
        
        fig, ax = plt.subplots()
        ax.plot(x * 10**9, lumins, c='k')   # plot the blackbody curve of the star
        ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
        ax.set_ylabel(r"Monochromatic Luminosity (W/nm)")
        ax.set_ylim(ymin=0); ax.set_xlim(xmin=0)
        
        if visible == True:     # plot the visible spectrum under the blackbody curve
            spectrum = np.linspace(1, 1000, 1000)
            colourmap = plt.get_cmap('Spectral_r')  # spectral_r is the visible spectrum going from blue -> red 
            normalize = colors.Normalize(vmin=380, vmax=750) # normalize spectral_r to the wavelength of the visible spectrum
            for i in range(len(spectrum) - 1): # iterate over blocks in the domain, and plot the colour for that block
                where = [True if 380 <= x <= 750 else False for x in [spectrum[i], spectrum[i + 1]]]
                ax.fill_between([spectrum[i], spectrum[i + 1]], [lumins[i], lumins[i + 1]], where=where, 
                                color=colourmap(normalize(spectrum[i])), alpha=0.3)
        if markers == True:     # plot markers for each of the luminosity band values given to the user
            ax.scatter(np.array([440, 500, 700]), self.bandlumin, color=['b', 'g', 'r'])
        
    
    def get_star_lumin(self):
        return self.luminosity
    def get_star_mass(self):
        return self.mass
    def get_star_temp(self):
        return self.temperature
    def get_star_radius(self):
        return self.radius
    def get_star_BandLumin(self):
        return self.bandlumin
    
        

class Galaxy(object):
    def __init__(self, species, position, population, radius, cartesian=False, BHcluster=True, darkmatter=True):
        '''
        Parameters
        ----------
        species : str
        position : 3-tuple/list/np.array
            if cartesian == False, position = [equatorial angle, polar angle, radius (distance away)]
            if cartesian == True, position = [x, y, z]
        '''
        self.darkmatter = darkmatter
        self.BHcluster = BHcluster
        self.species = species
        self.population = population
        self.radius = radius
        if cartesian:
            self.cartesian = position
            self.spherical = self.cartesian_to_spherical(position[0], position[1], position[2])
        else:
            self.spherical = position
            self.cartesian = self.spherical_to_cartesian(position[0], position[1], position[2])
        self.starpositions, self.stars, self.rotation = self.generate_galaxy()
        self.starmasses = [star.get_star_mass() for star in self.stars]
        self.blackhole = BlackHole(sum(self.starmasses), self.species, self.radius, 1)
        starorbitradii = [self.starpositions[0] - self.cartesian[0], 
                          self.starpositions[1] - self.cartesian[1], 
                          self.starpositions[2] - self.cartesian[2]]
        self.starorbits = self.star_orbits(starorbitradii[0], starorbitradii[1], starorbitradii[2])
        self.starvels, self.ObsStarVels = self.rotation_vels()
        
    def galaxyrotation(self, angle, axis):
        '''Rotate a point in cartesian coordinates about the origin by some angle along the specified axis. 
        The rotation matrices were taken from https://stackoverflow.com/questions/34050929/3d-point-rotation-algorithm
        
        Parameters
        ----------
        angle : float
            An angle in radians.
        axis : string
            The axis to perform the rotation on. Must be in ['x', 'y', 'z'].
        
        Returns
        -------
        numpy array
            The transformation matrix for the rotation of angle 'angle'. This output must be used as the first argument within "np.dot(a, b)"
            where 'b' is an 3 dimensional array of coordinates.
        '''
        if axis == 'x':
            m = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            return m
        elif axis == 'y':
            m = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            return m
        else:
            m = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            return m
    
    def generate_spiral(self, population, radius):
        '''Barred spiral galaxies generated using Fermat's spiral, and standard spiral galaxies generated using Archimedean spiral. 
        Returns
        -------
        x, y, z : np.array (x3)
            cartesian coordinates of each star in the galaxy
        colours : np.array
            Each element is an [R, G, B] value of each star to put into a matplotlib figure
        scales : np.array
            an array of floats which dictates how large a star appears on a matplotlib figure
        stars : np.array
            an array of Star objects, for each star in the galaxy
        '''
        # first step is to define the index to use in multiple data tables, based on the galaxy type:
        speciesindex = {"S0":0, "Sa":1, "Sb":2, "Sc":3, "SBa":4, "SBb": 5, "SBc":6}
        # galaxy radii need to be divided by certain numbers so that they behave as expected, with the divisor dependent on galaxy type
        SpiralRadiiDiv = [None, 15, 7, 2.1, 3.7, 3, 2.3]      #radius divisors (unitless)
        # for barred galaxies, the extent of the bar is different for different types of galaxy
        barradii = [0, 0, 0, 0, 0.3, 0.4, 0.5]  # bar radius as proportion of galaxy radius
        # next is the angular extents for each of the spiral arms in the form of [lower, upper] angle (radians)
        wrap = [[None, None], [0.9, 4 * np.pi], [0.7, 2 * np.pi], [0.2, 0.8 * np.pi], 
                [np.pi / 2.1, 3 * np.pi], [np.pi / 2.1, 2 * np.pi], [np.pi / 2.1, 1.15 * np.pi]]
        
        #now to actually grab the parameters for the galaxy type in question:
        mult, spiralwrap = [param[speciesindex[self.species]] for param in [SpiralRadiiDiv, wrap]]
        
        #[disk, bulge, bar, spiral] populations as a proportion of total galaxy star population
        regionpops = [[0.7, 0.2, 0, 0.01, 0.09],           #S0
                       [0.45, 0.2, 0, 0.15, 0.2],        #Sa
                       [0.45, 0.2, 0, 0.15, 0.2],        #Sb
                       [0.5, 0.2, 0, 0.1, 0.2],          #Sc
                       [0.3, 0.15, 0.2, 0.15, 0.2],      #SBa
                       [0.25, 0.15, 0.25, 0.15, 0.2],    #SBb
                       [0.4, 0.2, 0.2, 0.1, 0.1]]        #SBc
        
        #now to turn those population proportions into actual populations, given the current galaxy type
        diskpop, bulgepop, barpop, youngpop, oldpop = [int(prop * population) for prop in regionpops[speciesindex[self.species]]]
        spiralpop = youngpop + oldpop
        bulgeradius = radius / 10
        
        diskdists = np.random.exponential(radius / 4, size=diskpop)
        
        theta = np.random.uniform(0, 2*np.pi, diskpop)
        
        #this defines the disk star positions
        diskx = np.cos(theta) * diskdists
        disky = np.sin(theta) * diskdists
        diskz = np.zeros(diskpop) + 0.02 * radius * np.random.randn(diskpop)
        diskstars = self.generate_stars("disk", diskpop)
        diskcolours = [star.get_star_colour() for star in diskstars]
        diskscales = [star.get_star_scale() for star in diskstars]
        
        #this defines the bulge star positions
        # bulgedists = np.random.weibull(1.5 * bulgeradius, size = bulgepop) * np.random.normal(1, 0.05, bulgepop)
        bulgedists = np.random.exponential(bulgeradius/1.3, bulgepop) * np.random.normal(1, 0.05, bulgepop)
        theta = np.random.uniform(0, 2*np.pi, bulgepop)
        phi = np.random.uniform(-1, 1, bulgepop)
        phi = np.arccos(phi)
        
        bulgeR = bulgeradius * bulgedists**(1/3)    #bulgedists was meant to be RVs between 0 and 1, but the mult makes up for it
        bulgex = bulgeR * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, bulgepop))
        bulgey = bulgeR * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, bulgepop))
        distanceflat = (1 / radius) * np.sqrt(bulgex**2 + bulgey**2)     #this makes the z lower for stars further from the center
        bulgez = (0.83 * bulgeR * (np.cos(phi) + np.random.normal(0, 0.1, bulgepop))) * 0.9**distanceflat
        
        # bulgex = np.cos(theta) * bulgedists
        # bulgey = np.sin(theta) * bulgedists
        # bulgez = np.random.normal(0, 1.4/3 * bulgeradius, bulgepop)
        bulgestars = self.generate_stars("bulge", bulgepop)
        bulgecolours = [star.get_star_colour() for star in bulgestars]
        bulgescales = [star.get_star_scale() for star in bulgestars]
        
        
        if self.species[:2] == "SB":    #this will create the bar, given that the galaxy is a barred type
            barradius = barradii[speciesindex[self.species]] * radius
            barx = np.random.normal(0, 0.07 * barradius, barpop)
            bary = barradius * (np.geomspace(0.3, 1.1, barpop) * np.random.choice([-1, 1], barpop) + np.random.normal(0, 0.1, barpop))
            barz = np.random.normal(0, 0.05 * barradius, barpop)
            barstars = self.generate_stars("bulge", barpop)
            barcolours = [star.get_star_colour() for star in barstars]
            barscales = [star.get_star_scale() for star in barstars]
            bulgex = np.append(bulgex, barx); bulgey = np.append(bulgey, bary); bulgez = np.append(bulgez, barz)
            bulgecolours = np.append(bulgecolours, barcolours, axis=0)
            bulgescales = np.append(bulgescales, barscales, axis=0)
            bulgestars = np.append(bulgestars, barstars, axis=0)
        
        if self.BHcluster == True:
            BHx, BHy, BHz, BHcolours, BHscales, BHstars = self.generate_BHcluster()
            bulgex = np.append(bulgex, BHx); bulgey = np.append(bulgey, BHy); bulgez = np.append(bulgez, BHz)
            bulgecolours = np.append(bulgecolours, BHcolours, axis=0)
            bulgescales = np.append(bulgescales, BHscales, axis=0)
            bulgestars = np.append(bulgestars, BHstars, axis=0)
        
        # initialise some lists
        spiralx, spiraly, spiralz, spiralcolours, spiralscales, spiralstars = [], [], [], np.empty((0,3)), [], []
        
        if mult != None:          # time to generate spiral structure
            lower, upper = spiralwrap
            # youngpop, oldpop = int(spiralpop / 2), int(spiralpop / 2)
            youngstars = ["ys", youngpop, 0, 0.04, 0.01, 0.005, 10000, 6000] #[pop, lag, scatter, scatter2, zscatter, tempmean, tempshift]
            oldstars = ["os", oldpop, 0.2, 0.08, 0.015, 0.01, 4000, 1000]
            spiralpopulations = [youngstars, oldstars]
            
            for [region, pop, lag, scatter, scatter2, zscatter, tempmean, tempshift] in spiralpopulations:
                if speciesindex[self.species] >= 5:
                    spiralangle = np.geomspace(lower, upper, pop)
                else:
                    spiralangle = np.linspace(lower, upper, pop)
                reflect = np.random.choice([-1, 1], pop)
                power = 1/2 if self.species[:2] == "SB" else 1
                x = (radius / mult) * (spiralangle**power * np.cos(spiralangle + lag)  * np.random.normal(1, scatter, pop) * reflect + np.random.normal(0, scatter2, pop))
                y = (radius / mult) * (spiralangle**power * np.sin(spiralangle + lag) * np.random.normal(1, scatter, pop) * - reflect + np.random.normal(0, scatter2, pop))
                z = np.random.normal(0, zscatter * radius, pop)
                stars = self.generate_stars(region, pop)
                colours = [star.get_star_colour() for star in stars]
                scales = [star.get_star_scale() for star in stars]
                spiralx = np.append(spiralx, x); spiraly = np.append(spiraly, y); spiralz = np.append(spiralz, z)
                spiralcolours = np.append(spiralcolours, colours, axis=0)
                spiralscales = np.append(spiralscales, scales, axis=0)
                spiralstars = np.append(spiralstars, stars, axis=0)
        else:
            theta = np.random.uniform(0, 2*np.pi, spiralpop)
            x = np.cos(theta) * radius/1.5 * np.random.normal(1, 0.1, spiralpop)
            y = np.sin(theta) * radius/1.5 * np.random.normal(1, 0.1, spiralpop)
            z = np.zeros(spiralpop) + 0.02 * radius * np.random.randn(spiralpop)
            stars = self.generate_stars("disk", spiralpop)
            colours = [star.get_star_colour() for star in stars]
            scales = [star.get_star_scale() for star in stars]
            spiralx = np.append(spiralx, x); spiraly = np.append(spiraly, y); spiralz = np.append(spiralz, z)
            spiralcolours = np.append(spiralcolours, colours, axis=0)
            spiralscales = np.append(spiralscales, scales, axis=0)
            spiralstars = np.append(spiralstars, stars, axis=0)
                
        x = np.append(diskx, np.append(bulgex, spiralx)); y = np.append(disky, np.append(bulgey, spiraly)); z = np.append(diskz, np.append(bulgez, spiralz))
        colours = np.append(diskcolours, np.append(bulgecolours, spiralcolours, axis=0), axis=0)
        scales = np.append(diskscales, np.append(bulgescales, spiralscales, axis=0), axis=0)
        stars = np.append(diskstars, np.append(bulgestars, spiralstars, axis=0), axis=0)
        return x, y, z, colours, scales, stars
    
    def generate_elliptical(self, population, radius):
        '''Some guidance was taken from https://itecnote.com/tecnote/python-sampling-uniformly-distributed-random-points-inside-a-spherical-volume/
        Returns
        -------
        x, y, z : np.array (x3)
            cartesian coordinates of each star in the galaxy
        colours : np.array
            Each element is an [R, G, B] value of each star to put into a matplotlib figure
        scales : np.array
            an array of floats which dictates how large a star appears on a matplotlib figure
        stars : np.array
            an array of Star objects, for each star in the galaxy
        '''
        centralpop = int(0.2 * population); spherepop = int(0.8 * population)
        
        centralradius = radius / 6
        
        ellipsoid_mult = (1 - float(self.species[1]) / 10) # this makes later type ellipticals flatter (oblate)
        
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        
        spheredists = np.random.exponential(0.4, spherepop)
        centraldists = np.random.exponential(1/5, centralpop)
        centralR = centralradius * centraldists**(1/3)
        sphereR = radius * spheredists**(1/3)
        
        centralx = centralR * (np.cos(theta[:centralpop]) * np.sin(phi[:centralpop]) + np.random.normal(0, 0.1, centralpop))
        centraly = centralR * (np.sin(theta[:centralpop]) * np.sin(phi[:centralpop]) + np.random.normal(0, 0.1, centralpop))
        centralz = centralR * (np.cos(phi[:centralpop]) + np.random.normal(0, 0.05, centralpop))
        
        centralstars = self.generate_stars("bulge", centralpop)
        centralcolours = [star.get_star_colour() for star in centralstars]
        centralscales = [star.get_star_scale() for star in centralstars]
        
        if self.BHcluster == True:
            BHx, BHy, BHz, BHcolours, BHscales, BHstars = self.generate_BHcluster()
            centralx = np.append(centralx, BHx); centraly = np.append(centraly, BHy); centralz = np.append(centralz, BHz)
            centralcolours = np.append(centralcolours, BHcolours, axis=0)
            centralscales = np.append(centralscales, BHscales, axis=0)
            centralstars = np.append(centralstars, BHstars, axis=0)
        
        spherex = sphereR * (np.cos(theta[centralpop:]) * np.sin(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))
        spherey = sphereR * (np.sin(theta[centralpop:]) * np.sin(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))
        distanceflat = (1 / radius) * np.sqrt(spherex**2 + spherey**2)
        spherez = (sphereR * (np.cos(phi[centralpop:]) + np.random.normal(0, 0.1, spherepop))) * ellipsoid_mult**distanceflat
        
        spherestars = self.generate_stars("disk", spherepop)
        spherecolours = [star.get_star_colour() for star in spherestars]
        spherescales = [star.get_star_scale() for star in spherestars]
        
        x = np.append(centralx, spherex); y = np.append(centraly, spherey); z = np.append(centralz, spherez)
        colours = np.append(centralcolours, spherecolours, axis=0)
        scales = np.append(centralscales, spherescales, axis=0)
        
        stars = np.append(centralstars, spherestars, axis=0)
        return x, y, z, colours, scales, stars
    
    def generate_BHcluster(self):
        ''' Generate a cluster of stars close to the central black hole of a galaxy. The method for doing this is
        functionally identical to generating stars in an E0 galaxy (uniformly distributed stars in the volume of a sphere)
        '''
        population = 20
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        
        dists = np.random.exponential(0.4, population)
        radius = 0.1
        R = radius * dists**(1/3)
        x = R * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, population))
        y = R * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, population))
        z = R * (np.cos(phi) + np.random.normal(0, 0.05, population))
        
        stars = self.generate_stars("ys", population)
        colours = [star.get_star_colour() for star in stars]
        scales = [star.get_star_scale() for star in stars]
        return x, y, z, colours, scales, stars
    
    def generate_galaxy(self):
        '''Generate random stars according to species type of galaxy. 
        
        Returns
        -------
        numpy array (x4)
            Cartesian coordinates [x, y, z] of each of the stars in this galaxy, as well as an array of colours for each star. 
        '''
        population, radius = self.population, self.radius
        
        if self.species[0] == 'S':  #spiral galaxy
            x, y, z, colours, scales, stars = self.generate_spiral(population, radius)
        else:        #elliptical galaxy
            x, y, z, colours, scales, stars = self.generate_elliptical(population, radius)               
        
        points = np.array([x, y, z])
        phi = np.random.uniform(0, 2*np.pi, 3)
        
        #rotate the galaxy randomly
        points = np.dot(self.galaxyrotation(phi[0], 'x'), points)
        points = np.dot(self.galaxyrotation(phi[1], 'y'), points)
        points = np.dot(self.galaxyrotation(phi[2], 'z'), points)
        x0, y0, z0 = self.cartesian
        x, y, z = points[0] + x0, points[1] + y0, points[2] + z0  #move the galaxy away from the origin to its desired position
        return [x, y, z, colours, scales], stars, phi
        
    def get_stars(self):
        return self.starpositions
    def get_blackhole(self):
        return self.blackhole
    
    def star_orbits(self, x, y, z):
        ''' Finds the radius of the orbit of each star. 
        Returns
        -------
        radii : np.array
        '''
        radii = np.zeros(len(x))
        for i in range(len(x)):
            radii[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        return radii
    
    def generate_stars(self, region, n):
        '''Generates n Star objects according to the region of the galaxy.
        Parameters
        ----------
        region : str
            The region of the galaxy (e.g. young spiral, bulge, etc)
        n : int
            The number of stars to generate.
        
        Returns
        -------
        stars : list of n Star objects
        '''
        proportions = {"ys":[0.82, 0.1, 0.07, 0.01],    # [Main sequence, giants, supergiants, white dwarfs]
                       "os":[0.79, 0.15, 0.03, 0.03],
                       "disk":[0.9, 0.05, 0.02, 0.03], 
                       "bulge":[0.8, 0.1, 0.04, 0.06]}
        probs = proportions[region]     # obtain population probability for this region
        choice = []
        val = np.random.uniform(0, 1, n)
        for i in range(n):
            if val[i] <= probs[0]:
                choice.append("MS")
            elif val[i] <= probs[1] + probs[0]:
                choice.append("Giant")
            elif val[i] <= probs[2] + probs[1] + probs[0]:
                choice.append("SupGiant")
            else:
                choice.append("WDwarf")
        stars = [Star(region, species) for species in choice]
        return stars
    
    def rotation_vels(self):
        ''' Simulates orbit velocities of stars given their distance from the galactic center.
        If the galaxy has dark matter (self.darkmatter == True), then extra mass will be added according to the 
        Navarro-Frenk-White (NFW) dark matter halo mass profile. 
        TO DO: implement different dark matter halo properties for each galaxy type
        Returns
        -------
        np.array:
            2 element numpy array, with each element corresponding to:
                1. vel = the newtonian rotation velocities
                2. darkvel = rotation velocities including dark matter
            if self.darkmatter == False, then darkvel is an array of zeros
        '''
        if self.darkmatter == True:     # time to initialise dark matter properties 
            density = 0.01 # solar masses per cubic parsec
            scalerad = 150  # parsec
            Rs = scalerad * 3.086 * 10**16  # convert scalerad to meters
            p0 = density * (1.988 * 10**30 / (3.086 * 10**16)**3) # convert density to kg/m^3
            darkMass = lambda r: p0 / ((r / Rs) * (1 + r / Rs)**2) * (4 / 3 * np.pi * r**3)   # NFW dark matter profile (density * volume)
            
        G = 6.67 * 10**-11
        BHmass = self.blackhole.get_BH_mass() * 1.988 * 10**30
        
        masses, orbits = self.starmasses, self.starorbits
        # now, create an array that stores the mass and orbital radius of each star in the form of [[m1, r1], [m2,r2], ...]
        MassRadii = np.array([[masses[i] * 1.988 * 10**30, orbits[i] * 3.086 * 10**16] for i in range(len(masses))])
        vel = np.zeros(len(MassRadii)); darkvel = np.zeros(len(MassRadii))  # initialise arrays to store velocities in
        for i in range(len(MassRadii)):
            R = MassRadii[i, 1] 
            # now to sum up all of the mass inside the radius R
            M = sum([MassRadii[n, 0] if MassRadii[n, 1] < R else 0 for n in range(len(MassRadii))]) + BHmass
            vel[i] = (np.sqrt(G * M / R) / 1000)    # calculate newtonian approximation of orbital velocity
            if self.darkmatter == True:
                M += darkMass(R)    # add the average mass of dark matter inside the radius R
                darkvel[i] = (np.sqrt(G * M / R) / 1000)    # newtonian approximation, now including dark matter
        
        velarray = np.array([vel, darkvel]) * np.random.normal(1, 0.01, len(vel))
        

        # now to calculate the direction of the velocity to display the radial component to the observer
        x, y, z, _, _ = self.starpositions
        
        # now we need to transform the galaxy back to the origin with no rotation
        x, y, z = x - self.cartesian[0], y - self.cartesian[1], z - self.cartesian[2]
        points = np.array([x, y, z])
        phi = self.rotation
        
        # rotate galaxy in the reverse order and opposite direction as initially
        points = np.dot(self.galaxyrotation(-phi[2], 'z'), points)
        points = np.dot(self.galaxyrotation(-phi[1], 'y'), points)
        points = np.dot(self.galaxyrotation(-phi[0], 'x'), points)
        
        x, y, z = points
        if self.species[0] == "S":  # spiral galaxy! explanation in the comment block below :)
            theta = np.arctan2(y, x)
            direction = np.array([np.sin(theta), -np.cos(theta), np.random.normal(0, 0.05, len(theta))])
        #         _______                +y             +y|  /
        #         \   _  \               |                | /  \ theta
        # galaxy->/  /_\  \      -x  ____|____ +x         |/____] +x
        #         \  \_/   \             |
        #          \_____  /             |
        #                \/              -y
        # taking the arctan of y/x coordinates of stars gives clockwise circular motion about the galactic center
        # the proportion of motion in the [x, y, z] directions can then be calculated by:
        #     x => sin(theta), since we want theta angles between 0 and pi to have positive x-motion
        #     y => -cos(theta), since we want theta angles between -pi/2 and pi/2 to have negative y-motion
        #     z => normal(0, 0.05) since we want there to be negligible, but random z motion
        else:   # elliptical galaxy! explanation in the comment block below
            direction = np.array([np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))])
            for i in range(len(x)):
                xprop = np.random.uniform(-1, 1)
                yprop = np.random.uniform(-1, 1)
                while xprop**2 + yprop**2 > 1:
                    yprop = np.random.uniform(-1, 1)
                zprop = np.sqrt(1 - (xprop**2 + yprop**2))  # 1 = x**2 + y**2 + z**2 => z = sqrt(1 - x**2 - y**2)
                direction[0, i] = xprop; direction[1, i] = yprop; direction[2, i] = zprop
        # the squares of the directional velocity components must add up to one: 1 = xprop**2 + yprop**2 + zprop**2
        # so, we can randomly sample xprop and yprop (between -1 and 1 so that the velocity has random xy direction), 
        # making sure that the sum of their squares is not greater than one. Then, we can subtract the sum of their squares from
        # 1 to find the z component. All of this together gives more or less random direction to the stars about the galactic center. 

        direction = np.dot(self.galaxyrotation(phi[0], 'x'), direction)     # rotate the velocity vectors in the same way as before
        direction = np.dot(self.galaxyrotation(phi[1], 'y'), direction)
        direction = np.dot(self.galaxyrotation(phi[2], 'z'), direction)

        x, y, z, _, _ = self.starpositions  # getting the xyz again is cheaper than doing the rotations again
        
        velprops = np.zeros(len(x))
        for i in range(len(direction[0, :])):
            vector = direction[:, i]    # velocity vector "v"
            coord = np.array([x[i], y[i], z[i]])    # distance vector "d"
            velprops[i] = np.dot(vector, coord) / np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)      # dot product: (v dot d) / ||d||
            # the dot product above gets the radial component of the velocity (thank you Ciaran!! - linear algebra is hard)

        VelObsArray = velarray * velprops   # multiply the actual velocities by the line of sight proportion of the velocity magnitude
        return velarray, VelObsArray
    
    def plot_RotCurve(self, newtapprox=False, observed=False):
        ''' Produces a rotation curve of this galaxy. If the galaxy has dark matter and the user opts to display the newtonian
        approximation (curve based on visible matter), then two curves are plotted. 
        Parameters
        ----------
        newtapprox : bool
            whether to plot the newtonian approximation of the rotation curve (curve based on visible matter)
        observed : bool
            whether to plot the data that an observer would see (accounting for doppler shift)
        '''
        fig, ax = plt.subplots()
        if self.darkmatter == True:
            ax.scatter(self.starorbits, self.starvels[1], s=0.5, label="With Dark Matter")  # plot the dark matter curve data
            if observed == True:
                ax.scatter(self.starorbits, self.ObsStarVels[1], s=0.5, label="Observed")   # plot the data that the observer would see
            if newtapprox == True:
                ax.scatter(self.starorbits, self.starvels[0], s=0.5, label="Newtonian Approximation") # plot the newtonian approx as well
                if observed == True:
                    ax.scatter(self.starorbits, self.ObsStarVels[0], s=0.5, label="Observed")   # and plot the newtonian approx that the observer would see
                ax.legend()
        else: 
            ax.scatter(self.starorbits, self.starvels[0], s=0.5)    # plot the newtonian data
        
        ax.set_xlabel("Orbital Radius (pc)"); ax.set_ylabel("Orbital Velocity (km/s)")
        ax.set_ylim(ymin=0); ax.set_xlim(xmin=-0.1)
        
    def plot_doppler(self, fig, ax, cbar_ax, blackhole=True):
        ''' Plots the stars locations (similar to plot_2d), with colours indicating the stars radial velocity (line of sight motion)
        Positive velocities indicate motion away, negative towards. 
        I recommend initialising the fig, ax and cbar_ax in this way:
            fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1]})
        Since multiple instances of this function may be called onto the same fig/ax, the colour bar updates on each call, 
        first taking the data on ax and then merging the old and new data into one dataset in order to update the colourbar accordingly. 
        Help was gotten from https://stackoverflow.com/questions/33336343/recover-data-from-matplotlib-scatter-plot
        and also https://stackoverflow.com/questions/40614177/how-to-get-a-list-of-collections-on-a-matplotlib-figure
        Parameters
        ----------
        fig : matplotlib figure object
        ax : matplotlib axes
            the main axes to plot the stars on
        cbar_ax : matplotlib axes
            the secondary axes to plot the colourbar onto. I recommend this be >20 times thinner than the main axes
        blackhole : bool
            Whether or not the plot
        '''
        x, y, z, colours, scales = self.starpositions
        equat, polar, radius = self.cartesian_to_spherical(x, y, z)
        
        if self.darkmatter == True:
            starredshift = self.ObsStarVels[1]
        else:
            starredshift = self.ObsStarVels[0]
        
        # firstly, get the scatter data in the axes. If there is no data, then it will be a blank list which is no problem.
        data = [ax.collections[i].get_offsets().data for i in range(len(ax.collections))]
        # first data point for each ax addition is the star coords
        # the second data point are the star velocities
        # the third points are black hole locations, which have no use for altering the colourbar
        coords = [data[i] for i in np.arange(0, len(ax.collections), 3)] 
        speeds = [data[i] for i in np.arange(1, len(ax.collections), 3)] 
        
        # coords and speeds are inherently messy, so need to take data point from each nested array and add them to a neater array
        x, y, v = [], [], []
        for element in coords:  
            for coord in element:
                x.append(coord[0])
                y.append(coord[1])
        for element in speeds:
            for speed in element:
                v.append(speed[1])

        x = np.append(np.array(x), equat)   # merge the old and new data
        y = np.append(np.array(y), polar)
        v = np.append(np.array(v), starredshift)
        
        minvel = min(v); maxvel = max(v)
        if maxvel < -minvel:    # this conditional normalises the colourbar such that v=0 is in the middle of the max and min vel
            maxvel = -minvel
        else:
            minvel = -maxvel
    
        cm = plt.cm.get_cmap('bwr')     # blue => white => red colourmap
        red = ax.scatter(x, y, c=v, vmin=minvel, vmax=maxvel, cmap=cm , marker='.', s=0.5)  # note the colourmap for the redshift amount
        ax.scatter(np.zeros(len(v)), v, s=0)  # plots the speeds as 'nothing', so that speed values may be gathered on future calls of this function
        
        cbar = fig.colorbar(red, cax=cbar_ax)   # apply the colourbar to the cbar axes.
        cbar.set_label('Radial Velocity (km/s)', rotation=90)

        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')
        ax.set_aspect(1)    # sets it to be twice as wide as high, so that angular ratios are preserved
        fig.tight_layout()
        ax.invert_yaxis()
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
        
        if (self.blackhole != None) and (blackhole == True):    # plots the black hole if there is one, and if the user wants it
            BHequat, BHpolar, distance = self.spherical
            BHcolour = self.blackhole.get_BH_colour()
            BHscale = self.blackhole.get_BH_scale() / (0.05 * distance)
            ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale)
        else:
            ax.scatter(0, 0, s=0)   # plot 'nothing' so that the function works as intended
            
    def plot_HR(self, isoradii=False, xunit="temp", yunit="BolLum"):
        '''Plots a Colour-Magnitude (HR) diagram for this galaxy.     
        Parameters
        ----------
        isoradii : bool
            whether or not to plot constant radius lines on top of the HR diagram
        xunit : str
            One of {temp, colour, both}, which chooses what to put on the x-axis. "both" corresponds to temp on the bottom, colour on top
        yunit : str
            One of {BolLum, VLum, AbsMag, VMag, BolLumMag, bothV}, which chooses what to plot on the y-axis. 
            Bol-Mag corresponds to bolometric luminosity on the left y, and absolute magnitude on the right y
            bothV corresponds to V Band luminosity on the left y, V absolute mag on the right y
        Returns
        -------
        matplotlib figure
            The HR diagram. 
        '''
        fig, ax = plt.subplots()
        
        BolLum = [star.get_star_lumin() for star in self.stars]
        
        if (xunit in ["colour", "both"]) or (yunit in ["VLum", "VMag", "bothV"]):
            starBandLums = np.array([star.get_star_BandLumin() for star in self.stars])
            
            #this calculates the luminosity of the Sun at 500nm - same method as in Star.generate_BandLumin()
            c, h, k = 299792458, 6.626 * 10**-34, 1.38 * 10**-23
            planck = lambda x: ((2 * h * c**2) / x**5) * (1 / (np.exp(h * c / (x * k * 5778)) - 1)) * 10**-9
            solar500 = 4 * np.pi**2 * (696540000)**2 * planck(500 * 10**-9)
            
            starVLum = starBandLums[:, 1] / (solar500)    # get the 500nm luminosity of the star in solar units
            starBV = np.log10(starBandLums[:, 1] / starBandLums[:, 0]) # calculated as V - B, but is actually B - V due to the minus signs in their magnitude formulae
            if yunit in ["VMag", "bothV"]:
                mult = (3.828 * 10**26) / (3.0128 * 10**28)
                vmags = np.array([-2.5 * np.log10(lumin * mult) for lumin in starVLum])
        
        if yunit in ["AbsMag", "BolLumMag"]:
            mult = (3.828 * 10**26) / (3.0128 * 10**28)     # solar lum / 0-point lum on the mag scale. 
            BolMags = np.array([-2.5 * np.log10(lumin * mult) for lumin in BolLum])
        
        if xunit != "colour":
            temps = [star.get_star_temp() for star in self.stars]
            
        colours = self.starpositions[3]
        
        # now to decide what the x and y axis values are (and their alternate axes) given user input
        if xunit in ["temp", "both"]:
            xvals = temps; xlabel = "Temperature (K)"
            if xunit == "both":
                xlabel2 = r"Colour (B $-$ V)"
        else:
            xvals = starBV; xlabel = r"Colour (B $-$ V)"
        if yunit in ["BolLum", "AbsMag", "BolLumMag"]:
            if yunit == "AbsMag":
                yvals = BolMags; ylabel = r"Absolute Magnitude $M_{bol}$"
            else:
                yvals = BolLum; ylabel = r"Luminosity ($L / L_\odot$)"
            if yunit == "BolLumMag":
                yval2 = BolMags; ylabel2 = r"Absolute Magnitude $M_{bol}$"
        else:
            if yunit == "VMag":
                yvals = vmags; ylabel = r"V-Band Absolute Magnitude ($M_V$)"
            else:
                yvals = starVLum; ylabel = r"V-Band Luminosity ($L_V / L_{V, \odot}$)"
            if yunit == "bothV":
                yval2 = vmags; ylabel2 = r"V-Band Absolute Magnitude ($M_V$)"
        
        ax.scatter(xvals, yvals, color=colours, s=0.2)
        
        if xunit == "both":
            def TempVsColour(x, a, b, c, d, g):
                ''' A polynomial fit for temperature vs colour (B - V)
                '''
                return a * (1 / (b * x + c))**d + g
            
            # use scipy curve_fit to find a polynomial fit for temperature in terms of B - V colour
            fit, cov = opt.curve_fit(TempVsColour, starBV, temps, [4430, 1.6, 0.35, 0.58, -1930])
            
            ## uncomment the below if you want to calibrate the B - V colour and temperature fit
            # fitfit, fitax = plt.subplots()
            # fitax.scatter(starBV, temps)
            # x = np.linspace(min(starBV), max(starBV), 101)
            # y = func(x, fit[0], fit[1], fit[2], fit[3], fit[4])
            # fitax.plot(x, y)
            
            ax2 = ax.twiny()    # produce alternate x-axis on the top
            colourx = np.array([-0.2 + (n * 0.2) for n in range(10)]) # choose B - V values to plot
            tempx = TempVsColour(colourx, fit[0], fit[1], fit[2], fit[3], fit[4])   # calculate the temp for each colour
            ax2.scatter(np.log10(tempx), np.array([1 for i in range(10)]), alpha=0)  # plot them so that they show up on the plot
            ax2.set_xlabel(xlabel2);
            #now to define the ticks and make their labels in terms of the colours
            ax2.set_xticks(np.log10(tempx)); ax2.set_xticklabels([round(num, 1) for num in colourx], fontsize=6)
            ax2.minorticks_off(); ax2.invert_xaxis()
            
        if yunit in ["BolLumMag", "bothV"]:
            ax3 = ax.twinx()
            ax3.scatter(xvals, yval2, color=colours, s=0.2)
            ax3.set_ylabel(ylabel2); ax3.invert_yaxis()
            
        if xunit in ["temp", "both"]:
            ax.invert_xaxis(); ax.set_xscale('log')
            ax.set_xticks([10**4, 2 * 10**4, 5 * 10**3, 2 * 10**3])     # set custom temperature labels
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())    # remove scientific notation

        if yunit in ["BolLum", "BolLumMag", "VLum", "bothV"]:
            ax.set_yscale('log')
        else:
            ax.invert_yaxis()
            
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_facecolor('k')
        
        if (isoradii == True) and (xunit in ["temp", "both"]) and (yunit in ["BolLum", "BolLumMag"]):
            textcolour = [0.7, 0.7, 0.7]
            solarradius = 696340000     #initialise variables
            solarlum = 3.828 * 10**26
            sigma = 5.67037 * 10**-8
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()  #get the current figure bounds so that we don't alter it
            x = np.linspace(xmin, xmax, 2)
            # now to plot the isoradii lines on the HR diagram
            for power in np.arange(-3, 5):
                y = (4 * np.pi * (solarradius * 10.0**power)**2 * sigma * x**4) / solarlum
                ax.plot(x, y, linewidth=0.6, linestyle='--', color=textcolour)
                if power == 0:
                    text = "$R_\odot$"
                elif power == 1:
                    text = "$10R_\odot$"
                else:
                    text = f"$10^{{{power}}} R_\odot$"
                if ymin < max(y) < ymax:    #this makes sure that text doesn't show up outside of the plot bounds
                    ax.text(max(x), max(y), text, color=textcolour, rotation=-23, fontsize=8)
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)    #make sure the figure bounds dont change from before
                
    
    def plot_2d(self, fig, ax, spikes=False, radio=False):
        '''Plots the Galaxy onto predefined matplotlib axes in terms of its equatorial and polar angles. 
        
        Parameters
        ----------
        fig : matplotlib.figure
        ax : matplotlib.axes 
            A predefined matplotlib axes that has been defined by "fig, ax = plt.subplots()"
        spikes : bool
            Whether to show diffraction spikes for bright stars.
        
        Returns
        -------
        No returns, but adds the current Galaxy instance to the matplotlib axes. 
        '''
        x, y, z, colours, scales = self.starpositions
        equat, polar, radius = self.cartesian_to_spherical(x, y, z)
        
        if self.blackhole != None:
            BHequat, BHpolar, distance = self.spherical
            BHcolour = self.blackhole.get_BH_colour()
            BHscale = self.blackhole.get_BH_scale() / (0.05 * distance)
            if BHscale > 2.5: 
                spikesize = BHscale / 2
                ax.errorbar(BHequat, BHpolar, yerr=spikesize, xerr=spikesize, ecolor=BHcolour, fmt='none', elinewidth=0.3, alpha=0.5)
            ax.scatter(BHequat, BHpolar, color=BHcolour, s=BHscale)
        
        scales = scales / (0.05 * radius)
        if spikes == True:
            brightequat, brightpolar, brightscale, brightcolour = [], [], [], np.empty((0, 3))
            for i, scale in enumerate(scales):
                if scale > 2.5:
                    brightequat += [equat[i]]
                    brightpolar += [polar[i]]
                    brightscale = brightscale + [scale / 4]
                    brightcolour = np.append(brightcolour, [colours[i]], axis=0)
            ax.errorbar(brightequat, brightpolar, yerr=brightscale, xerr=brightscale, ecolor=brightcolour, fmt='none', elinewidth=0.3)
        scales = [4 if scale > 4 else abs(scale) for scale in scales]
        ax.scatter(equat, polar, s=scales, c=colours)
        ax.set_xlim(0, 360); ax.set_ylim(0, 180)
        ax.set_facecolor('k')
        ax.set_aspect(1)
        fig.tight_layout()
        ax.invert_yaxis()
        ax.set_xlabel("Equatorial Angle (degrees)")
        ax.set_ylabel("Polar Angle (degrees)")
        if radio == True:
            self.plot_radio_contour(ax)
    
    def plot_3d(self, ax, camera=False):
        '''Plots the Galaxy onto predefined 3D matplotlib axes. 
        
        Parameters
        ----------
        ax : matplotlib.axes 
            A predefined matplotlib axes that has been defined by "ax = fig.add_subplot(projection='3d')", 
            where fig is defined by "fig = plt.figure()"
        camera : bool
            whether or not to show a little green pyramid at the origin (0, 0, 0) showing the direction of the camera in the 2d plot
        
        Returns
        -------
        No returns, but adds the current Galaxy instance to the matplotlib axes. 
        '''
        x, y, z, colours, scales = self.starpositions
        ax.scatter(-x, -y, -z, s=scales, c=colours) #need to plot the flipped coordinates for some reason? need to do this to match up with the 2d plot.
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # # uncomment the lines below if you want to calibrate galaxy radii, etc
        # ax.view_init(elev=90, azim=0)
        # ax.set_facecolor([0.4, 0.4, 0.4])
        
        # uncomment the lines below for a pretty view
        ax.set_facecolor('k')
        
        if camera == True:  # plot a silly little camera showing in which direction and position the observer is looking from in the 2d plot
            ax.scatter(0, 0, 0, c='g', s=60, alpha=0.9) #plots the main camera part
            equat = np.array([-30, -30, 30, 30]) + 180
            polar = np.array([-30, 30, 30, -30]) + 90
            distance = self.radius / 2
            equat = np.radians(equat); polar = np.radians(polar)
            x = distance * np.cos(equat) * np.sin(polar)
            y = distance * np.sin(equat) * np.sin(polar)
            z = distance * np.cos(polar)
            ax.scatter(x, y, z, c='g', s=40, alpha=0.9)
            x, y, z = np.append(x, x[0]), np.append(y, y[0]), np.append(z, z[0])
            ax.plot(x, y, z, c='g', linewidth=1)
            for i in range(4):
                ax.plot([0, x[i]], [0, y[i]], [0, z[i]], c='g', linewidth=1)
    
    def plot_radio3d(self):
        ''' Plots the faux radio emission jets/lobes (the 3D scattered points), with the origin (galaxy SMBH) at the origin. 
        '''
        x, y, z, radius = self.blackhole.get_BH_radio()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, s=0.8)
        ax.set_xlim(-radius, radius); ax.set_ylim(-radius, radius), ax.set_zlim(-radius, radius)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    
    def plot_radio_contour(self, ax, plot=True, scatter=False, data=False):
        ''' Plot the radio contours of the SMBH emission onto a 2D sky plot. 
        Parameters
        ----------
        ax : matplotlib axes object
        plot : bool
            Whether to actually plot the contour
        scatter : bool
            Whether to overlay the raw scatter data for calibration purposes
        data : bool
            Whether to return the area density data for the contours
        Returns (if data=True)
        -------
        equatbins, polarbins : numpy arrays (1xN)
            The equatorial and polar coordinates of the contour density bins. 
        density : numpy array (NxN)
            The number count of scatter particles per equat/polar bin. 
        '''
        x, y, z, radius = self.blackhole.get_BH_radio()
        phi = self.rotation
        points = np.array([x, y, z])
        points = np.dot(self.galaxyrotation(phi[0], 'x'), points) # radio scatter is centered at the origin, 
        points = np.dot(self.galaxyrotation(phi[1], 'y'), points) # so we need to rotate it in the same way as the galaxy was
        points = np.dot(self.galaxyrotation(phi[2], 'z'), points)
        x, y, z = points
        x, y, z = x + self.cartesian[0], y + self.cartesian[1], z + self.cartesian[2] # and now translate it to where the galaxy is
        equat, polar, distance = self.cartesian_to_spherical(x, y, z)
            
        extent = [[min(equat) - 3, max(equat) + 3], [min(polar) - 3, max(polar) + 3]]   # this is so that the edge of the contours aren't cut off
        density, equatedges, polaredges = np.histogram2d(equat, polar, bins=len(equat)//50, range=extent, density=False)
        equatbins = equatedges[:-1] + (equatedges[1] - equatedges[0]) / 2   # this fixes the order of the bins, and centers the bins at the midpoint
        polarbins = polaredges[:-1] + (polaredges[1] - polaredges[0]) / 2

        density = density.T      # take the transpose of the density matrix
        density = scipy.ndimage.zoom(density, 2)    # this smooths out the data so that it's less boxy and more curvey
        equatbins = scipy.ndimage.zoom(equatbins, 2)
        polarbins = scipy.ndimage.zoom(polarbins, 2)
        # density = scipy.ndimage.gaussian_filter(density, sigma=1)  # this smooths the area density even moreso (not necessary, but keeping for posterity)

        if plot == True:    # plot the contour
            levels = [2, 3, 4, 5, 6, 10, 15]    # having the contour levels start at 2 removes the noise from the smoothing - important!!
            ax.contour(equatbins, polarbins, density, levels, corner_mask=True)     # plot the radio contours
            ax.set_ylim(0, 180); ax.set_xlim(0, 360)
            ax.invert_yaxis();
            if scatter == True:     # plot the actual scattered points on top of the contour - mainly just for calibration
                ax.scatter(equat, polar, s=0.5)
        if data == True:
            return equatbins, polarbins, density
            # equat/polar are 1xN matrices, whereas density is a NxN matrix.  
    
    def cartesian_to_spherical(self, x, y, z):
        ''' Converts cartesian coordinates to spherical ones (formulae taken from wikipedia) in units of degrees. 
        Maps polar angle to [0, 180] with 0 at the north pole, 180 at the south pole. 
        Maps azimuthal (equatorial) angle to [0, 360], with equat=0 corresponding to the negative x-axis, equat=270 the positive y-axis, etc
        Azimuthal (equat) angles reference (rotates anti-clockwise):
            equat = 0 or 360 -> -ve x-axis (i.e. y=0)
            equat = 90 -> -ve y-axis (x=0)
            equat = 180 -> +ve x-axis (y=0)
            equat = 270 -> +ve y-axis (x=0)
        Parameters
        ----------
        x, y, z : numpy array
            x, y, and z cartesian coordinates
        
        Returns
        -------
        equat, polar, radius : numpy array
            equatorial and polar angles (in degrees), and radius from origin
        '''
        radius = np.sqrt(x**2 + y**2 + z**2)
        equat = np.arctan2(y, x)    #returns equatorial angle in radians, maps to [-pi, pi]
        polar = np.arccos(z / radius)
        polar = np.degrees(polar)
        equat = np.degrees(equat)
        # now need to shift the angles
        if np.size(equat) != 1:
            equat = np.array([360 - abs(val) if val < 0 else val for val in equat])  #this reflects negative angles about equat=180
        else:   #same as above, but for a single element. 
            equat = 360 - abs(equat) if equat < 0 else equat
        return (equat, polar, radius)
    
    def spherical_to_cartesian(self, equat, polar, distance):
        '''
        Parameters
        ----------
        equat, polar, distance : numpy array
            equatorial and polar angles, as well as radial distance from the origin
        
        Returns
        -------
        x, y, z : numpy array
            Cartesian coordinates relative to the origin. 
        '''
        equat, polar = np.radians(equat), np.radians(polar)
        x = distance * np.cos(equat) * np.sin(polar)
        y = distance * np.sin(equat) * np.sin(polar)
        z = distance * np.cos(polar)
        return (x, y, z)
    
class GalaxyCluster(object):
    def __init__(self, position, population, cartesian=False, local=False):
        '''
        Parameters
        ----------
        position : 3-tuple/list/np.array
            if cartesian == False, position = [equatorial angle, polar angle, radius (distance away)]
            if cartesian == True, position = [x, y, z]
        local : bool
            Whether this is the local galaxy cluster (i.e. the one that the observer at the origin is in)
        '''
        self.local = local
        self.radius = 1000
        if cartesian:
            self.cartesian = position
            self.spherical = self.cartesian_to_spherical(position[0], position[1], position[2])
        else:
            self.spherical = position
            self.cartesian = self.spherical_to_cartesian(position[0], position[1], position[2])
        self.galaxies = self.generate_galaxies(population)
    
    def generate_galaxy(self, species, position, population, radius):
        return Galaxy(species, position, population, radius, cartesian=True)
    
    def generate_galaxies(self, population):
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        
        dists = np.random.exponential(0.4, population)
        R = self.radius * dists**(1/3)
        
        x = R * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, population)) + self.cartesian[0]
        y = R * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, population)) + self.cartesian[1]
        z = R * (np.cos(phi) + np.random.normal(0, 0.05, population)) + self.cartesian[2]
        
        # args = [('Sa', [x[i], y[i], z[i]], 600, 70) for i in range(len(x))]
        # print(args)
        # pool = Pool()
        # galaxies = pool.starmap(self.generate_galaxy, args)
        # pool.close()
        # pool.join()
        args = [['Sa', [x[i], y[i], z[i]], 500, 70] for i in range(len(x))]
        galaxies = []
        for i in range(len(x)):
            species, position, population, radius = args[i]
            print(species, position, population, radius)
            galaxies.append(Galaxy(species, position, population, radius, cartesian=True))
        return galaxies
            
    def cartesian_to_spherical(self, x, y, z):
        ''' Converts cartesian coordinates to spherical ones (formulae taken from wikipedia) in units of degrees. 
        Maps polar angle to [0, 180] with 0 at the north pole, 180 at the south pole. 
        Maps azimuthal (equatorial) angle to [0, 360], with equat=0 corresponding to the negative x-axis, equat=270 the positive y-axis, etc
        Azimuthal (equat) angles reference (rotates anti-clockwise):
            equat = 0 or 360 -> -ve x-axis (i.e. y=0)
            equat = 90 -> -ve y-axis (x=0)
            equat = 180 -> +ve x-axis (y=0)
            equat = 270 -> +ve y-axis (x=0)
        Parameters
        ----------
        x, y, z : numpy array
            x, y, and z cartesian coordinates
        
        Returns
        -------
        equat, polar, radius : numpy array
            equatorial and polar angles (in degrees), and radius from origin
        '''
        radius = np.sqrt(x**2 + y**2 + z**2)
        equat = np.arctan2(y, x)    #returns equatorial angle in radians, maps to [-pi, pi]
        polar = np.arccos(z / radius)
        polar = np.degrees(polar)
        equat = np.degrees(equat)
        # now need to shift the angles
        if np.size(equat) != 1:
            equat = np.array([360 - abs(val) if val < 0 else val for val in equat])  #this reflects negative angles about equat=180
        else:   #same as above, but for a single element. 
            equat = 360 - abs(equat) if equat < 0 else equat
        return (equat, polar, radius)
    
    def spherical_to_cartesian(self, equat, polar, distance):
        '''
        Parameters
        ----------
        equat, polar, distance : numpy array
            equatorial and polar angles, as well as radial distance from the origin
        
        Returns
        -------
        x, y, z : numpy array
            Cartesian coordinates relative to the origin. 
        '''
        equat, polar = np.radians(equat), np.radians(polar)
        x = distance * np.cos(equat) * np.sin(polar)
        y = distance * np.sin(equat) * np.sin(polar)
        z = distance * np.cos(polar)
        return (x, y, z)
    
def plot_all_dopplers(galaxies):
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30,1]})
    for galaxy in galaxies:
        galaxy.plot_doppler(fig, ax, cbar_ax, blackhole=True)
def plot_all_2d(galaxies, spikes=False, radio=False):
    fig, ax = plt.subplots()
    for galaxy in galaxies:
        galaxy.plot_2d(fig, ax, spikes=spikes, radio=radio)
        
def main():
    # galaxy = Galaxy('SBb', (0,500,100), 1000, 100, cartesian=True)
    # galaxy = Galaxy('Sc', (180, 90, 500), 1000, 70)
    # galaxy2 = Galaxy('E0', (104, 131, 500), 1000, 100)
    # galaxy3 = Galaxy('Sc', (110, 128, 1000), 1000, 50)
    # galaxies = [galaxy]
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # galaxy.plot_3d(ax, camera=False)
    
    # galaxy.plot_radio3d()
    
    # fig, ax = plt.subplots()
    # galaxy.plot_radio_contour(ax)
    # galaxy.plot_RotCurve(newtapprox=False, observed=True)
    # galaxy.plot_HR(isoradii=True, xunit="both", yunit="BolLumMag")
    # ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(-15, 15)
    # ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(-10, 10)

    # plot_all_dopplers(galaxies)
    # plot_all_2d(galaxies, spikes=True, radio=True)
    # galaxy.plot_HR(isoradii=True)
    # fig, ax = plt.subplots()
    # galaxy.plot_2d(fig, ax, spikes=True, radio=True)
    # galaxy2.plot_2d(fig, ax, spikes=True, radio=True)
    # galaxy3.plot_2d(fig, ax, spikes=True, radio=True)
    
    cluster = GalaxyCluster((180, 90, 2000), 5)
    print(cluster.galaxies)
    plot_all_2d(cluster.galaxies)

    
if __name__ == "__main__":
    main()