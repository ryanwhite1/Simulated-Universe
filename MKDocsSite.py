# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:08:43 2023

@author: ryanw
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

class MkSite(object):
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
        with open(self.directory + "/datasetMain.md", "w") as primary:
            primary.write("---\ntitle: Data Home\n---")
            
            if self.proj in ["Cube", "DivCube"]:
                primary.write("## Images from the Wide Field Cameras")
                primary.write("Click on the images for the data on that part of the sky!\n")
                
                messages = ["'Right' joins the left of this image, and 'Left' the right.",
                            "'Left' joins the top of this image, 'Front' the right, 'Right' the bottom, and 'Back' the left.", 
                            "'Left' joins the left of this image, and 'Right' the right.", 
                            "'Back' joins the left of this image, and 'Front' the right.", 
                            "'Front' joins the left of this image, and 'Back' the right.", 
                            "'Right' joins the top of this image, 'Front' the right, 'Left' the bottom, and 'Back' the left."]
                
                for i, direction in enumerate(["Back", "Bottom", "Front", "Left", "Right", "Top"]):
                    # show the image for each direction
                    primary.write(f"### {direction} Camera Image")
                    primary.write(f'![{direction}]({direction}/{direction}.png)')
                    primary.write(messages[i])
                
                self.cubemap_image()
                primary.write("### Cubemap Image Alignment")
                primary.write('![cubemap](Cubemap.png)')
                    
            else:
                primary.write("## Image from the All Sky Widefield Camera")
            
            if variables == True:
                # need to add the variable star data folder to a .zip archive in order to make it downloadable.
                # the below code does this. 
                import shutil
                shutil.make_archive(self.directory + "/Variable_Star_Data", 'zip', self.directory + "/Variable_Star_Data")
                
                primary.write("## Variable Star Data")
                primary.write("Some stars in the sky were found to change in apparent brightness over time, usually following a periodic trend.")
                for filename in os.listdir(self.directory + "/Variable_Star_Data"):
                    # i want to show the first variable star lightcurve that has an image.
                    if filename[-3:] == "png":
                        primary.write(f'![variable](Variable_Star_Data/{filename})')
                        primary.write(f"Example of a variable star lightcurve - star {filename[:-4]}")
                        break
                primary.write("Units of the variable data are:")
                primary.write("Measurement | Unit \n--- | ---\n Time | hours\n Normalised Flux | unitless")
                primary.write("Uncertainties in this data (one standard deviation) are:")
                primary.write("Measurement | Uncertainty \n--- | ---\n Time | 0.3 hours\n Normalised Flux | 1.5%")
                primary.write('[Download a compressed .zip file of all of the variable star lightcurve data.](Variable_Star_Data.zip)')
                
            if flashes == True:
                primary.write("## X-Ray All-Sky Camera Data")
                primary.write("The X-Ray camera did not detect any steady sources. It did detect, however, a number of extremely ")
                primary.write("short X-Ray flashes coming from various parts of the sky. Each flash lasted 51 milliseconds, but ")
                primary.write("we see that the number of photons from each flash is far from the same.")
                primary.write("Here is a list of the flashes detected, with their approx. positions and number of photons detected. ")
                if self.proj in ["Cube", "DivCube"]:
                    primary.write("Positions are given by where they would appear in the relevant wide-field camera image. ")
                primary.write("Positions are only accurate to 0.05 degrees (one standard deviation). <p>")
                primary.write("The X-Ray camera is sensitive to burts of more than 174 photons only. <p>")
                if self.proj in ["AllSky", "Both"]:
                    primary.write('[Text file of the below data](AllSky_Flash_Data.csv)')
                    primary.write(self.create_md_table(self.directory + "/AllSky_Flash_Data.csv"))
                elif self.proj in ["Cube", "DivCube"]:
                    primary.write('[Text file of the below data](Flash_Data.csv)')
                    primary.write(self.create_md_table(self.directory + "/Flash_Data.csv"))
            
    def face_index(self):
        ''' Creates and saves one or more index.html files to represent data based on the data projection. Will create
        one index file if using the All-Sky projection, and 6 index.html files in their respective folders if using a 
        cubemapped dataset. 
        
        ### INCOMPLETE ###
        '''
        if self.proj in ["Cube", "DivCube"]:
            for i, direction in enumerate(["Back", "Bottom", "Front", "Left", "Right", "Top"]):
                with open(self.directory + f"/{direction}/datasetSubMain.md", "w") as secondary:
                    secondary.write(f'## Data from the {direction} Wide-Field Camera')
                    secondary.write(f'![{direction}-pic]({direction}.png)')
                    
                    if self.proj == "Cube":
                        secondary.write("### Stellar Objects in this image")
                        secondary.write("**Note:** positive radial velocities indicated objects moving away from us.")
                        secondary.write("Blue flux is measured at 440nm, green at 500nm, and red at 700nm.")
                        secondary.write("Units are:")
                        secondary.write("Measurement | Unit \n--- | ---\n Position | degrees\n Flux | W/nm/m^2^\n Parallax | arcsec\n Radial Velocity | km/s\n Variable | unitless (=1 if variable, =0 if not)")
                        secondary.write("Uncertainties in this data (one standard deviation figures) are:")
                        secondary.write("Measurement | Uncertainty \n--- | ---\n Flux | 1%\n Position | 0.0001 degrees\n Parallax | 0.001 arcseconds\n Radial Velocity | 0.03km/s")
                        secondary.write('[Click here to download the star data in .csv format, suitable for loading into python or matlab!](Star_Data.csv)')
                        
                        secondary.write("### Distant Galaxies in this image")
                        secondary.write("**Note:** positive radial velocities indicated objects moving away from us.")
                        secondary.write("Blue flux is measured at 440nm, green at 500nm, and red at 700nm.")
                        secondary.write("Units are:")
                        secondary.write("Measurement | Unit \n--- | ---\n Position | degrees\n Flux | W/nm/m^2^\n Size | arcseconds\n Radial Velocity | km/s")
                        secondary.write("Uncertainties in this data (one standard deviation figures) are :")
                        secondary.write("Measurement | Uncertainty \n--- | ---\n Position | 0.0001 degrees \nFluxes | 1% \nSize | 10% \n Radial Velocity | 0.1km/s")
                        secondary.write('[Click here to download the distant galaxy data in .csv format, suitable for loading into python or matlab!](Distant_Galaxy_Data.csv)')
                    elif self.proj == "DivCube":
                        secondary.write("The above image is divided into a 6x6 grid, totalling 36 images. The subdivided images, and links to data contained within those images are available below!")
                        for j, Y in enumerate(["1", "2", "3", "4", "5", "6"]):
                            for k, X in enumerate(["A", "B", "C", "D", "E", "F"]):
                                secondary.write(f'[{X+Y}]({X+Y}/datasetSubdivision.md)')
                                with open(self.directory + f"/{direction}/{X+Y}/datasetSubdivision.md", "w") as subindex:
                                    subindex.write(f'## Data from the {direction} Wide-Field Camera, Division {X+Y}')
                                    subindex.write(f'[subdivision-pic]({X}{Y}_{direction}.png)')
                                    
                                    subindex.write("### Stellar Objects in this image")
                                    subindex.write("**Note:** positive radial velocities indicated objects moving away from us.")
                                    subindex.write("Blue flux is measured at 440nm, green at 500nm, and red at 700nm.")
                                    subindex.write("Units are:")
                                    subindex.write("Measurement | Unit \n--- | ---\n Position | degrees\n Flux | W/nm/m^2^\n Parallax | arcsec\n Radial Velocity | km/s\n Variable | unitless (=1 if variable, =0 if not)")
                                    subindex.write("Uncertainties in this data (one standard deviation figures) are:")
                                    subindex.write("Measurement | Uncertainty \n--- | ---\n Flux | 1%\n Position | 0.0001 degrees\n Parallax | 0.001 arcseconds\n Radial Velocity | 0.03km/s")
                                    subindex.write('[Click here to download the star data in .csv format, suitable for loading into python or matlab!](Star_Data.csv)')
                                    
                                    subindex.write("### Distant Galaxies in this image")
                                    subindex.write("**Note:** positive radial velocities indicated objects moving away from us.")
                                    subindex.write("Blue flux is measured at 440nm, green at 500nm, and red at 700nm.")
                                    subindex.write("Units are:")
                                    subindex.write("Measurement | Unit \n--- | ---\n Position | degrees\n Flux | W/nm/m^2^\n Size | arcseconds\n Radial Velocity | km/s")
                                    subindex.write("Uncertainties in this data (one standard deviation figures) are :")
                                    subindex.write("Measurement | Uncertainty \n--- | ---\n Position | 0.0001 degrees \nFluxes | 1% \nSize | 10% \n Radial Velocity | 0.1km/s")
                                    subindex.write('[Click here to download the distant galaxy data in .csv format, suitable for loading into python or matlab!](Distant_Galaxy_Data.csv)')
                    
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
    
    def create_md_table(self, datafile, delimiter=','):
        ''' Returns the markdown code for a basic table, with data from the datafile file.
        Parameters
        ----------
        datafile : str
            The location and name of the data file to be converted to html format.
        delimiter : str
            The spacing between elements within the datafile.
        Returns
        -------
        datastring : str
            A string which contains all of the data of the datafile, in a markdown format.
        '''
        data = pd.read_csv(datafile, delimiter=delimiter)
        col_names = list(data.columns.values)
        datastring = ''
        for index, row in data.iterrows():
            if index == 0:
                for i in range(len(col_names)):
                    datastring += f"{col_names[i]} | "
                datastring += '\n'
                for i in range(len(col_names)):
                    datastring += "--- | "
                datastring += '\n'
            else:
                for i in range(len(col_names)):
                    datastring += f'{row[col_names[i]]} | '
                datastring += '\n'
        return datastring
    
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
        
        
        
        

