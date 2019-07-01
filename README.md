# facial-recognition-system

**DOPPELGÄNGER: Facial Recognition System** is a program that determines the celebrity or politician that a person most closely resembles using the FaceNet CNN and images from IMDB and Wikipedia.

## Repository Contents

The following 11 files and 3 folders are included in this repository:

  1.	**FaceNet_Model/** - A folder containing the Keras FaceNet model and weights
  2.	**Photo_Dataframes/** - A folder containing the metadata and vector embeddings of all faces in the database
  3.	**image_data/** - A folder containing the user images and imdb/wikipedia images (for size purposes a download link is included)
  4.	**.gitignore** - Contains the files excluded from this repository (images that collectively take up too much space and test notebooks)
  5.	**EDA.ipynb** - Contains both Exploratory Data Analysis and the methodology for eliminating irrelevant images 
  6.	**FaceNet Obtain Embeddings.ipynb** - A notebook for obtaining the facial embeddings vector from the FaceNet model
  7.	**README.md** - This document
  8.	**doppelganger_finder.py** - The main file; run this document with the requirements file listed to use the program
  9.	**environment.yml** - A file for replicating the environment required to run **doppelganger_finder.py**
  10.	**face_detection_functions.py** - A python script containing seven functions for the detection, extraction and comparison of faces using the FaceNet model and MTCNN library 
  11.	**imdb_metadata_cleanup.ipynb** - The metadata for the images from IMDB (conversion from matlab)
  12.	**photo_program.py** - A python file containing the function to take a picture with a device's camera using OpenCV
  13.	**requirements.txt** - The requirements file for running **doppelganger_finder.py**
  14.	**wiki_metadata_cleanup.ipynb** - The metadata for the images from Wikipedia (conversion from matlab)

# The Data

The images and metadata come from ETH Zürich's Computer Vision Lab. There are 460,723 images from IMDB and 62,328 images from Wikipedia that correspond to +70k individuals. The data was collected in 2015 by Rasmus Rothe, Radu Timofte, and Luc Van Gool for their papers on age detection ([Deep expectation of real and apparent age from a single image without facial landmarks](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)). The faces are already cropped by the ETH Zürich team using OpenCV's cascade classifier. I downloaded the two cropped faces folders which total 8 GB.

I did not include the images in this repository because of size limitations. The **imdb_data/** and **wiki_data/** subfolders are stored on my local machine in the **image_data/** folder. Each of these folders contain a **.mat** file of metadata and 100 subfolders containing images of cropped faces.

The notebooks **imdb_metadata_cleanup.ipynb** and **wiki_metadata_cleanup.ipynb** contain the process of converting the matlab files to pandas dataframes. I save each dataframe as a **.csv** file in the **Photo_Dataframes/** folder. Both the imdb and wikipedia metadata contain similar columns. The first five rows of the combined data frame is below:

![Metadata Data Frame Head](/image_data/Images_for_ReadMe/dataframe_header.png)

# EDA/Image Selection

The notebooks **imdb_metadata_cleanup.ipynb** and **wiki_metadata_cleanup.ipynb**

# The Model

# The Program
