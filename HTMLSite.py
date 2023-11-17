# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:19:58 2023

@author: ryanw
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

class HTMLSite(object):
    ''' Should be used in conjunction with the UniverseSim object to generate a (very) basic html site which provides a more 
    user friendly interface for analysing data. 
    '''
    def __init__(self, directory, proj="Cube", flashes=True, variables=True):
        ''' Initialise and create the html files.
        Parameters
        ----------
        directory : str
            The main data directory of this UniverseSim dataset
        proj : str
            One of {'AllSky', 'Cube', 'Both', 'DivCube'} to determine output of images/data
        flashes : bool
            If we've saved the flash events. Will generate a section in the top-level index file about xray flashes
        variables : bool
            If data for variable stars has been created and saved prior to this class being initialised.
        
        '''
        self.directory = directory
        self.proj = proj
        self.top_index(flashes=flashes, variables=variables)
        self.face_index()
        
    
    def top_index(self, flashes=True, variables=True):
        ''' Creates and saves a top-level .html file that shows some data, with links to download more data!
        Parameters
        ----------
        flashes : bool
            If we've saved the flash events. Will generate a section in the top-level index file about xray flashes
        variables : bool
            If data for variable stars has been created and saved prior to this class being initialised.
        '''
        with open(self.directory + "/index.html", "w") as index:
            index.write("<html>\n<head>\n<title>Data Home</title>\n</head>\n<body>")
            
            if self.proj in ["Cube", "DivCube"]:
                index.write("<H1> Images from the Wide Field Cameras: </H1>")
                index.write("Click on the images for the data on that part of the sky!\n<hr>")
                
                messages = ["'Right' joins the left of this image, and 'Left' the right.",
                            "'Left' joins the top of this image, 'Front' the right, 'Right' the bottom, and 'Back' the left.", 
                            "'Left' joins the left of this image, and 'Right' the right.", 
                            "'Back' joins the left of this image, and 'Front' the right.", 
                            "'Front' joins the left of this image, and 'Back' the right.", 
                            "'Right' joins the top of this image, 'Front' the right, 'Left' the bottom, and 'Back' the left."]
                
                for i, direction in enumerate(["Back", "Bottom", "Front", "Left", "Right", "Top"]):
                    # show the image for each direction
                    index.write(f"<H2> {direction} Camera Image </H2>")
                    index.write(f'<A HREF="{direction}/index.html"><IMG SRC="{direction}/{direction}.png" WIDTH="600"></A><BR>')
                    index.write(messages[i] + "<p>\n<hr>")
                
                self.cubemap_image()
                index.write("<H3> Cubemap Image Alignment </H3>")
                index.write('<IMG SRC="Cubemap.png" WIDTH="600"><br><hr>')
                    
            else:
                index.write("<H1> Image from the All Sky Widefield Camera: </H1>")
            
            if variables == True:
                # need to add the variable star data folder to a .zip archive in order to make it downloadable.
                # the below code does this. 
                import shutil
                shutil.make_archive(self.directory + "/Variable_Star_Data", 'zip', self.directory + "/Variable_Star_Data")
                
                index.write("<H1> Variable Star Data </H1>")
                index.write("<p>Some stars in the sky were found to change in apparent brightness over time, usually following a periodic trend.<br><br>")
                for filename in os.listdir(self.directory + "/Variable_Star_Data"):
                    # i want to show the first variable star lightcurve that has an image.
                    if filename[-3:] == "png":
                        index.write(f'<IMG SRC="Variable_Star_Data/{filename}" WIDTH="500"><br>')
                        index.write(f"<p>Example of a variable star lightcurve - star {filename[:-4]} <br>")
                        break
                index.write("<p>Units of the variable data are:<ul>\n<li> Time: hours\n<li> Normalised flux: unitless</ul>")
                index.write("<p>Uncertainties in this data (one standard deviation) are:<ul>\n<li> Time: 0.3 hours\n<li> Norm. flux: 1.5%</ul>")
                index.write('<H3><A HREF="Variable_Star_Data.zip">Download a compressed .zip file of all of the variable star lightcurve data.</A></H3><br><hr>')
                
            if flashes == True:
                index.write("<H1> X-Ray All-Sky Camera Data </H1>")
                index.write("The X-Ray camera did not detect any steady sources. It did detect, however, a number of extremely ")
                index.write("short X-Ray flashes coming from various parts of the sky. Each flash lasted 51 milliseconds, but ")
                index.write("we see that the number of photons from each flash is far from the same. <p>")
                index.write("Here is a list of the flashes detected, with their approx. positions and number of photons detected. ")
                if self.proj in ["Cube", "DivCube"]:
                    index.write("Positions are given by where they would appear in the relevant wide-field camera image. ")
                index.write("Positions are only accurate to 0.05 degrees (one standard deviation). <p>")
                index.write("The X-Ray camera is sensitive to burts of more than 174 photons only. <p>")
                if self.proj in ["AllSky", "Both"]:
                    index.write('<H3><A HREF="AllSky_Flash_Data.csv">Text file of the below data</A></H3>')
                    index.write(self.create_html_table(self.directory + "/AllSky_Flash_Data.csv"))
                elif self.proj in ["Cube", "DivCube"]:
                    index.write('<H3><A HREF="Flash_Data.csv">Text file of the below data</A></H3>')
                    index.write(self.create_html_table(self.directory + "/Flash_Data.csv"))
                
            index.write("</body>")
            
    def face_index(self):
        ''' Creates and saves one or more index.html files to represent data based on the data projection. Will create
        one index file if using the All-Sky projection, and 6 index.html files in their respective folders if using a 
        cubemapped dataset. 
        
        ### INCOMPLETE ###
        '''
        if self.proj in ["Cube", "DivCube"]:
            for i, direction in enumerate(["Back", "Bottom", "Front", "Left", "Right", "Top"]):
                with open(self.directory + f"/{direction}/index.html", "w") as index:
                    index.write(f'<html>\n<head>\n<title>Data from the {direction} Wide-Field Camera </title>\n</head>')
                    index.write(f'<body>\n<IMG SRC="{direction}.png" WIDTH="600"><br><hr>')
                    
                    if self.proj == "Cube":
                        index.write("<H2> Stellar Objects in this image </H2>")
                        index.write("<p><b>Note:</b> positive radial velocities indicated objects moving away from us.</p>")
                        index.write("<p>Blue flux is measured at 440nm, green at 500nm, and red at 700nm.</p>")
                        index.write("<p>Units are:<ul>")
                        index.write("<li> Positions: degrees\n<li> Flux: W/nm/m<sup>2</sup>\n<li> Parallax: arcsec\n<li> Radial Velocity: km/s\n<li> Variable: unitless (=1 if variable, =0 if not)</ul>")
                        index.write("<p>Uncertainties are (one standard deviation figures):<ul>")
                        index.write("<li> Fluxes: 1% \n<li> Positions: 0.0001 degrees \n<li> Parallaxes: 0.001 arcseconds\n<li> Radial Velocities: 0.03km/s</ul>")
                        index.write('<A HREF="Star_Data.csv">Click here to download the star data in .csv format, suitable for loading into python or matlab!</A><br><hr>')
                        
                        index.write("<H2> Distant Galaxies in this image </H2>")
                        index.write("<p><b>Note:</b> positive radial velocities indicated objects moving away from us.</p>")
                        index.write("<p>Blue flux is measured at 440nm, green at 500nm, and red at 700nm.</p>")
                        index.write("<p>Units are:<ul>")
                        index.write("<li> Positions: degrees\n<li> Flux: W/nm/m<sup>2</sup>\n<li> Size: arcseconds\n<li> Radial Velocity: km/s</ul>")
                        index.write("<p>Uncertainties are (one standard deviation figures):<ul>")
                        index.write("<li> Fluxes: 1% \n<li> Positions: 0.0001 degrees \n<li> Sizes: 10% \n<li> Radial Velocities: 0.1km/s</ul>")
                        index.write('<A HREF="Distant_Galaxy_Data.csv">Click here to download the distant galaxy data in .csv format, suitable for loading into python or matlab!</A><br><hr>')
                    elif self.proj == "DivCube":
                        index.write("The above image is divided into a 6x6 grid, totalling 36 images. The subdivided images, and links to data contained within those images are available below! <br><hr>")
                        for j, Y in enumerate(["1", "2", "3", "4", "5", "6"]):
                            for k, X in enumerate(["A", "B", "C", "D", "E", "F"]):
                                index.write(f'<A HREF="{X+Y}/index.html"> {X+Y}</A>')
                                with open(self.directory + f"/{direction}/{X+Y}/index.html", "w") as subindex:
                                    subindex.write(f'<html>\n<head>\n<title>Data from the {direction} Wide-Field Camera, Division {X+Y} </title>\n</head>')
                                    subindex.write(f'<body>\n<IMG SRC="{X}{Y}_{direction}.png"><br><hr>')
                                    
                                    subindex.write("<H2> Stellar Objects in this image </H2>")
                                    subindex.write("<p><b>Note:</b> positive radial velocities indicated objects moving away from us.</p>")
                                    subindex.write("<p>Blue flux is measured at 440nm, green at 500nm, and red at 700nm.</p>")
                                    subindex.write("<p>Units are:<ul>")
                                    subindex.write("<li> Positions: degrees\n<li> Flux: W/nm/m<sup>2</sup>\n<li> Parallax: arcsec\n<li> Radial Velocity: km/s\n<li> Variable: unitless (=1 if variable, =0 if not)</ul>")
                                    subindex.write("<p>Uncertainties are (one standard deviation figures):<ul>")
                                    subindex.write("<li> Fluxes: 1% \n<li> Positions: 0.0001 degrees \n<li> Parallaxes: 0.001 arcseconds\n<li> Radial Velocities: 0.03km/s</ul>")
                                    subindex.write('<A HREF="Star_Data.csv">Click here to download the star data in .csv format, suitable for loading into python or matlab!</A><br><hr>')
                                    
                                    subindex.write("<H2> Distant Galaxies in this image </H2>")
                                    subindex.write("<p><b>Note:</b> positive radial velocities indicated objects moving away from us.</p>")
                                    subindex.write("<p>Blue flux is measured at 440nm, green at 500nm, and red at 700nm.</p>")
                                    subindex.write("<p>Units are:<ul>")
                                    subindex.write("<li> Positions: degrees\n<li> Flux: W/nm/m<sup>2</sup>\n<li> Size: arcseconds\n<li> Radial Velocity: km/s</ul>")
                                    subindex.write("<p>Uncertainties are (one standard deviation figures):<ul>")
                                    subindex.write("<li> Fluxes: 1% \n<li> Positions: 0.0001 degrees \n<li> Sizes: 10% \n<li> Radial Velocities: 0.1km/s</ul>")
                                    subindex.write('<A HREF="Distant_Galaxy_Data.csv">Click here to download the distant galaxy data in .csv format, suitable for loading into python or matlab!</A><br><hr>')
                            index.write("<br>")
                        index.write("</body>")
                    
    def create_html_table(self, datafile, delimiter=','):
        ''' Returns the html for a basic table, with data from the datafile file.
        Parameters
        ----------
        datafile : str
            The location and name of the data file to be converted to html format.
        delimiter : str
            The spacing between elements within the datafile.
        Returns
        -------
        html : str
            A string which contains all of the data of the datafile, in html format to be displayed on a webpage.
        '''
        data = pd.read_csv(datafile, delimiter=delimiter)
        html = data.to_html(index=False)
        return html
    
    def cubemap_image(self):
        ''' Creates and saves a basic image showing the alignment of the 6 faces of a cube in the UniverseSim cubemap orientation.
        '''
        # these points make up a cube. Each list within the big list are two points that draw out a line
        x = [[0, 4], [0, 4], [0, 0], [4, 4], [1, 2], [1, 2], [1, 1], [2, 2], [3, 3]]
        y = [[2, 2], [1, 1], [2, 1], [2, 1], [3, 3], [0, 0], [0, 3], [0, 3], [1, 2]]
        # the below are X,Y coords to place the text saying which face of the cube we're looking at
        positions = [[0.35, 1.5], [1.35, 1.5], [2.35, 1.5], [3.35, 1.5], [1.35, 2.5], [1.35, 0.5]]
        faces = ["Back", "Left", "Front", "Right", "Top", "Bottom"]

        fig, ax = plt.subplots()

        for i in range(len(x)): # now plot all of the lines
            ax.plot(x[i], y[i], c='k')
            ax.axis('off')
        for i in range(6): # and now put the text on the cube faces
            ax.text(positions[i][0], positions[i][1], faces[i])
        
        # finally, save the image in the main data directory
        fig.savefig(self.directory + '/Cubemap.png', bbox_inches='tight')
        plt.close()
        
        
        
        