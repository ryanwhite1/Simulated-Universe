# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:38:20 2022

@author: ryanw
"""
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import MiscTools as misc
from Galaxy import Galaxy
from DistantGalaxy import DistantGalaxy

class GalaxyCluster(object):
    ''' A cluster of Galaxy objects, all spherically distributed about a central position. Generates galaxy objects within this
    cluster.
    '''
    def __init__(self, position, population, cartesian=False, local=False, blackholes=True, darkmatter=True, complexity="Normal",
                 variable=[True, [24.6, "Tri", -6.5, 59], [40.7, "Saw", -14, 64], [75.6, "Sine", 17.9, 35.1]], rotvels="Normal"):
        ''' Generates a few/several Galaxy objects about a central position.
        Parameters
        ----------
        position : 3-tuple/list/np.array
            if cartesian == False, position = [equatorial angle, polar angle, radius (distance away)]
            if cartesian == True, position = [x, y, z]
        population : int
            The number of galaxies to generate in the cluster
        cartesian : bool
            True if the position argument is given in terms of cartesian (x, y, z) coordinates
        local : bool
            Whether this is the local galaxy cluster (i.e. the one that the observer at the origin is in)
        blackholes : bool
            If true, generates black holes in the galaxies
        darkmatter : bool
            If true, generates dark matter in the galaxies and in the cluster
        complexity : str
            One of {"Comprehensive", "Normal", "Basic", "Distant"} which dictates the population of the galaxies and the type. 
        variable : list
            The first element must be a bool, which decides whether or not to generate variability in some stars
            The second and third elements (and fourth [optional]) must be comprised of 
            [period, lightcurve type, gradient, yint], where the period is in hours (float) and the 
            lightcurve type is one of {"Saw", "Tri", "Sine"} (str), with gradient and yint being floats too. 
            See Universe.py -> "determine_variablestars" for more info
        rotvels : str
            One of {"Normal", "Boosted"}, which dictates whether rotation curves have arbitrarily boosted velocity 
            magnitudes (possibly accounting for interstellar gas/dust mass?).
        '''
        self.local = local
        self.clusterpop = population
        self.radius = 200 * population**(5/6)   # we need bigger clusters to have proportionally bigger radii to fit all of the galaxies in!
        self.blackholes = blackholes
        self.darkmatter = darkmatter
        self.complexity = complexity
        
        if cartesian:
            self.cartesian = position
            self.spherical = misc.cartesian_to_spherical(position[0], position[1], position[2])
        else:
            self.spherical = position
            self.cartesian = misc.spherical_to_cartesian(position[0], position[1], position[2])
        self.variable = variable
        self.rotvelMult = rotvels
        self.galaxies, self.galaxmasses, self.galaxorbits, self.galaxpositions = self.generate_galaxies(population)
        self.galaxvels, self.ObsGalaxVels, self.darkmattermass, self.directions = self.rotation_vels(mult=self.rotvelMult)
        
        if rotvels == "Boosted":
            self.clustermass = 4 * sum(self.galaxmasses) + self.darkmattermass
        elif rotvels == "Normal":
            self.clustermass = sum(self.galaxmasses) + self.darkmattermass
    
    def generate_galaxy(self, species, position, rotate, seed):
        ''' Generate a Galaxy class object.
        Parameters
        ----------
        species : str
            The type of galaxy to generate
        position : 3-tuple
            The xyz position of the galaxy (cartesian coords)
        rotate : bool
            If true, generates a galaxy with rotation. If false, no rotation (the local galaxy to the observer!)
        seed : int
            Random number generation seed for this galaxy (only required for non-distant galaxies)
        Returns
        -------
        Galaxy : Galaxy object
        '''
        if self.complexity == "Distant":
            return DistantGalaxy(species, position, cartesian=True, blackhole=self.blackholes, darkmatter=self.darkmatter)
        else:
            return Galaxy(species, position, cartesian=True, rotate=rotate, blackhole=self.blackholes, 
                          darkmatter=self.darkmatter, complexity=self.complexity, variable=self.variable, 
                          rotvels=self.rotvelMult, seed=seed)
    
    def generate_galaxies(self, population):
        ''' Uniformly distributes and generates galaxies within a sphere, with a central elliptical galaxy if the cluster
        population is high enough (>=5) and a cD galaxy if population is >=10.
        Parameters
        ----------
        population : int
            The number of galaxies to generate
        Returns
        -------
        galaxies : list
            list of Galaxy class objects that comprise the cluster
        galaxmasses : numpy array
            The mass of each galaxy in the cluster (including dark matter), in units of solar masses.
        orbitradii : numpy array
            The orbital radius of each galaxy from the center of the cluster, in units of parsec
        galaxpositions : numpy array
            Cartesian coordinates of each galaxy in the cluster (in relation to cluster center)
        '''
        seeds = np.random.randint(0, 1e5, population)
        theta = np.random.uniform(0, 2*np.pi, population)
        phi = np.random.uniform(-1, 1, population)
        phi = np.arccos(phi)
        
        # dists = np.random.exponential(0.4, population)
        dists = np.random.uniform(0, 1, population)
        R = self.radius * dists**(1/3)
        
        x = R * (np.cos(theta) * np.sin(phi) + np.random.normal(0, 0.1, population))
        y = R * (np.sin(theta) * np.sin(phi) + np.random.normal(0, 0.1, population))
        z = R * (np.cos(phi) + np.random.normal(0, 0.05, population))
        orbitradii = np.sqrt(x**2 + y**2 + z**2)    # orbit radius for each galaxy from the cluster center, in parsecs
        galaxpositions = np.array([x, y, z])
        
        # now to move the galaxies to their appropriate position in the sky
        x, y, z = x + self.cartesian[0], y + self.cartesian[1], z + self.cartesian[2]
        
        # determine the types of galaxies in this cluster
        species = self.species_picker(orbitradii, population)
        # print(species)
        if population >= 5:     # need a central elliptical galaxy, so generate n-1 galaxies
            args = [(species[i], [x[i], y[i], z[i]], True, seeds[i]) for i in range(len(x) - 1)]    # species type at galaxy location
        else:       # no central elliptical needed, so generate n galaxies
            args = [(species[i], [x[i], y[i], z[i]], True, seeds[i]) for i in range(len(x))]

        if population >= 10:
            args.insert(0, ('cD', self.cartesian, True, seeds[0]))  # insert a cD galaxy in the center of the cluster
        elif population >= 5:
            num = 9 - population    # more populous clusters will have a bigger, more spherical elliptical in their center
            args.insert(0, (f'E{num}', self.cartesian, True, seeds[0]))     # insert an elliptical galaxy in the center of the cluster
        
        if self.local == True:  # generate a spiral galaxy near the observer, with no rotation
            # choose spiral type - we dont want S0 because they have fewer variable stars (due to their position on HR diagrams)
            # also don't want SBc or Sc due to their wild shape
            localgalaxy = np.random.choice(['SBa', 'SBb', 'Sa', 'Sb'])   
            localdist = np.random.uniform(15, 50)   # we don't want the observer in the center of the galaxy, but also not outside of it
            localx, localy, localz = misc.spherical_to_cartesian(180, 90, localdist)
            args[-1] = (localgalaxy, [localx, localy, localz], False, seeds[-1])   # replace the last galaxy in the cluster with this galaxy, with no rotation!
            # galaxpositions[-1, :] = np.array([localx, localy, localz])
            galaxpositions[-1, 0] = localx
            galaxpositions[-1, 1] = localy
            galaxpositions[-1, 2] = localz
            orbitradii[-1] = np.sqrt(localx**2 + localy**2 + localz**2)
        # now, use multiprocessing to generate the galaxies in the cluster according to the arguments above and their positions
        if self.complexity not in ["Distant", "Basic"]:
            with Pool() as pool:
                galaxies = pool.starmap(self.generate_galaxy, args)
        else:   # distant galaxies actually generate faster without multiprocessing
            galaxies = []
            for arg in args:
                galaxies.append(self.generate_galaxy(arg[0], arg[1], arg[2], arg[3]))
        
        galaxmasses = np.zeros(len(galaxies))
        for i, galaxy in enumerate(galaxies):
            galaxmasses[i] = galaxy.galaxymass  # get the mass of each galaxy
            
        return galaxies, galaxmasses, orbitradii, galaxpositions
    
    def species_picker(self, orbitradii, population):
        ''' A function to determine the type of galaxies in the cluster, given their orbital radii from the cluster center
        and the cluster population. 
        Parameters
        ----------
        orbitradii : list or numpy array
            The radius of each orbit of the galaxies with the cluster, in units of parsec
        population : int 
            Population of galaxies in the cluster
        Returns
        -------
        types : list
            The species of each galaxy, in the same order as the galaxies in orbitradii
        '''
        if population == 1:
            types = ['S0']
        else:
            types = []
            for i in range(len(orbitradii)):
                prop = orbitradii[i] / self.radius  # determine how far from the center the galaxy is
                elliptcheck = np.random.uniform(0, 1)   # generate a RV to determine whether a galaxy is elliptical
                
                if population >= 5:     # dense cluster
                    # the below makes it more likely for ellipticals the larger the cluster pop is, up to a pop of 10 when it then has 
                    # constant probability
                    elliptical = True if elliptcheck <= (min(0.05 * population, 1) - prop) else False
                    spiral = not elliptical     # of course the galaxy can't be both a spiral and elliptical
                else:
                    elliptical = True if elliptcheck <= 0.1 else False  # about 10% chance of an elliptical galaxy outside of a dense cluster
                    spiral = not elliptical
    
                if spiral == True:
                    barcheck = np.random.uniform(0, 1)
                    barred = True if barcheck <= 0.7 else False     # 70% of spiral galaxies have central bars
                    if barred == True:
                        types.append(np.random.choice(['SBa', 'SBb', 'SBc']))   # choose a barred galaxy
                    else:
                        types.append(np.random.choice(['S0', 'Sa', 'Sb', 'Sc']))    # choose a non-barred galaxy
                else:   # elliptical == True:
                    n = int(7 * min(prop, 1))   # more likely for spherical ellipticals closer to the center of the cluster
                    types.append(f'E{n}')
        return types
    
    def rotation_vels(self, mult="Normal"):
        ''' Simulates orbit velocities of stars given their distance from the galactic center.
        If the galaxy has dark matter (self.darkmatter == True), then extra mass will be added according to the 
        Navarro-Frenk-White (NFW) dark matter halo mass profile. 
        Returns
        -------
        velarray : np.array
            2 element numpy array, with each element corresponding to:
                1. vel = the newtonian rotation velocities
                2. darkvel = rotation velocities including dark matter
            if self.darkmatter == False, then darkvel=vel
        VelObsArray : np.array
            Same format as velarray, but is the line-of-sight (radial) velocities as seen by the observer at the origin
        darkmattermass : float
            The mass of dark matter in 1.5x the cluster radius (maximum dist of a galaxy from the cluster center). Units are solar masses
        direction : numpy array
            The directions (as proportions of velocity magnitude in each cartesian coordinate axis) of galaxy motion
        '''
        if self.darkmatter == True:     # time to initialise dark matter properties 
            density = 0.00005 # solar masses per cubic parsec
            if mult == "Boosted":
                density *= 10
            scalerad = 1.3 * self.radius  # parsec
            Rs = scalerad * 3.086 * 10**16  # convert scalerad to meters
            p0 = density * (1.988 * 10**30 / (3.086 * 10**16)**3) # convert density to kg/m^3
            darkMass = lambda r: p0 / ((r / Rs) * (1 + r / Rs)**2) * (4 / 3 * np.pi * r**3)   # NFW dark matter profile (density * volume)
            
        G = 6.67 * 10**-11
        
        masses, orbits = self.galaxmasses, self.galaxorbits
        if mult == "Boosted":
            masses *= 4
        # now, create an array that stores the mass and orbital radius of each star in the form of [[m1, r1], [m2,r2], ...]
        MassRadii = np.array([[masses[i] * 1.988 * 10**30, orbits[i] * 3.086 * 10**16] for i in range(len(masses))])
        vel = np.zeros(len(MassRadii)); darkvel = np.zeros(len(MassRadii))  # initialise arrays to store velocities in
        for i in range(len(MassRadii)):
            R = MassRadii[i, 1] 
            # now to sum up all of the mass inside the radius R
            M = sum([MassRadii[n, 0] if MassRadii[n, 1] < R else 0 for n in range(len(MassRadii))])
            vel[i] = (np.sqrt(G * M / R) / 1000)    # calculate newtonian approximation of orbital velocity
            if self.darkmatter == True:
                M += darkMass(R)    # add the average mass of dark matter inside the radius R
                darkvel[i] = (np.sqrt(G * M / R) / 1000)    # newtonian approximation, now including dark matter
            else:
                darkvel[i] = vel[i]
                
        darkmattermass = darkMass(1.5 * max(MassRadii[:, 1])) if self.darkmatter == True else 0
        darkmattermass /= 1.988 * 10**30    # get the darkmatter mass in units of solar masses
        
        velarray = np.array([vel, darkvel]) * np.random.normal(1, 0.01, len(vel))

        # now to calculate the direction of the velocity to display the radial component to the observer
        x, y, z = self.galaxpositions
        
        # now we need to transform the galaxy back to the origin with no rotation
        x, y, z = x - self.cartesian[0], y - self.cartesian[1], z - self.cartesian[2]
        
        # now to generate random velocity directions for the galaxies
        direction = np.array([np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))])
        xprop = np.random.uniform(-1, 1, len(x))
        yprop = np.random.uniform(-1, 1, len(x))
        zprop = np.random.uniform(-1, 1, len(x))
        mult = np.sqrt(1 / (xprop**2 + yprop**2 + zprop**2))         # find a multiplier so that the sum of the squares is not greater than 1
        xprop *= mult; yprop *= mult; zprop *= mult
        direction[0, :] = xprop; direction[1, :] = yprop; direction[2, :] = zprop
        # the squares of the directional velocity components must add up to one: 1 = xprop**2 + yprop**2 + zprop**2
        # so, we can randomly sample xprop, yprop, and zprop (between -1 and 1 so that the velocity has random xyz direction), 
        # making sure that the sum of their squares is not greater than one. 
        # i.e., if we randomly sample xyz, we need that `mult^2 * (x^2 + y^2 + z^2) = 1 => mult = sqrt(1/(x^2 + y^2 + z^2))
        # then we replace x = mult * x, y = mult * y, z = mult * z, so that their sums actually do add to 1. 

        x, y, z = self.galaxpositions  # getting the xyz again is cheaper than doing operations again
        
        velprops = np.zeros(len(x))
        for i in range(len(direction[0, :])):
            vector = direction[:, i]    # velocity vector "v"
            coord = np.array([x[i], y[i], z[i]])    # distance vector "d"
            velprops[i] = np.dot(vector, coord) / orbits[i]      # dot product: (v dot d) / ||d||
            # the dot product above gets the radial component of the velocity (thank you Ciaran!! - linear algebra is hard)

        VelObsArray = velarray * velprops   # multiply the actual velocities by the line of sight proportion of the velocity magnitude
        return velarray, VelObsArray, darkmattermass, direction
    
    def plot_RotCurve(self, newtapprox=False, observed=False, save=False):
        ''' Produces a rotation curve of this galaxy. If the galaxy has dark matter and the user opts to display the newtonian
        approximation (curve based on visible matter), then two curves are plotted. 
        Parameters
        ----------
        newtapprox : bool
            whether to plot the newtonian approximation of the rotation curve (curve based on visible matter)
        observed : bool
            whether to plot the data that an observer would see (accounting for doppler shift)
        Returns
        -------
        fig : matplotlib figure object
            If save==True, returns the figure that the plot is on
        '''
        fig, ax = plt.subplots()
        if self.darkmatter == True:
            ax.scatter(self.galaxorbits, self.galaxvels[1], s=0.5, label="With Dark Matter")  # plot the dark matter curve data
            if observed == True:
                ax.scatter(self.galaxorbits, abs(self.ObsGalaxVels[1]), s=0.5, label="Observed")   # plot the data that the observer would see
            if newtapprox == True:
                ax.scatter(self.galaxorbits, self.galaxvels[0], s=0.5, label="Newtonian Approximation") # plot the newtonian approx as well
                if observed == True:
                    ax.scatter(self.galaxorbits, abs(self.ObsGalaxVels[0]), s=0.5, label="Observed")   # and plot the newtonian approx that the observer would see
                ax.legend()
        else: 
            ax.scatter(self.galaxorbits, self.galaxvels[0], s=0.5)    # plot the newtonian data
        
        ax.set_xlabel("Orbital Radius (pc)"); ax.set_ylabel("Orbital Velocity (km/s)")
        ax.set_ylim(ymin=0); ax.set_xlim(xmin=-0.1)
        
        if save:
            plt.close()
            return fig