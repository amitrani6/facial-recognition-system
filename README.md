# facial-recognition-system

The **DOPPELGÄNGER: Facial Recognition System** is a program that determines the celebrity or politician that a person most closely resembles using the FaceNet CNN and images from IMDB and Wikipedia.

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

The images and metadata come from ETH Zürich's Computer Vision Lab. There are 460,723 images from IMDB and 62,328 images from Wikipedia that correspond to +70k individuals. The data was collected in 2015 by Rasmus Rothe, Radu Timofte, and Luc Van Gool for their papers on age detection ([Deep expectation of real and apparent age from a single image without facial landmarks](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)). The faces are already cropped by the ETH Zürich team using a Deformable Parts Model (DPM) described in the paper [Face Detection Without The Bells And Whistles](http://rodrigob.github.io/documents/2014_eccv_face_detection_with_supplementary_material.pdf) (*by Markus Mathias, Rodrigo Benenson, Marco Pedersoli, and Luc Van Gool*). I downloaded the two cropped faces zip files which total 8 GB.

I did not include the images in this repository because of size limitations. The **imdb_data/** and **wiki_data/** subfolders are stored on my local machine in the **image_data/** folder. Each of these folders contain a **.mat** file of metadata and 100 subfolders containing images of cropped faces, which is how the two .tar files (*downloaded from ETH Zürich link above*) appear when they are unzipped.

The notebooks **imdb_metadata_cleanup.ipynb** and **wiki_metadata_cleanup.ipynb** contain the process of converting the matlab files to pandas dataframes. I save each dataframe as a **.csv** file in the **Photo_Dataframes/** folder. Both the imdb and wikipedia metadata contain similar columns. The first five rows of the combined data frame is below:

![Metadata Data Frame Head](/image_data/Images_for_ReadMe/dataframe_header.png)


**The Metadata Mainframe Information**

**Given**:

-**name**: Name of the person

-**dob**: date of birth (Matlab serial date number)

-**gender**: 0 for female and 1 for male, NaN if unknown

-**photo_taken**: year when the photo was taken

-**full_path**: path to the image file

-**face_location**: location of the face in the uncropped image.

-**face_score**: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image

-**second_face_score**: detector score of the face with the second highest score. I use this to remove group images by setting a threshold (explained below)

**Not Seen In The Above Dataframe**:

-**celeb_names (IMDB only)**: list of all celebrity names

-**celeb_id (IMDB only)**: imdb index of celebrity name (obsolete, this no longer corresponds to imdb profiles)

**Created**:

-**age_when_taken**: birthdate subtracted from 'photo_taken' **Not Seen In This Image**

# EDA/Image Selection

The **EDA.ipynb** notebook contains both the Exploratory Data Analysis and the methodology for eliminating images that are either missing metadata or have low face scores. There are 523,051 images in the initial dataset corresponding to +70k indivduals.

### Missing Names

I eliminated the 124 images with missing names as I would not be able to associate a user's face to them correctly.

### Birthday Conversion

![Age Distribution](/image_data/Images_for_ReadMe/Age.jpg)

The birthdays associated with each profile and image were webscraped off of IMDB and Wikipedia. If the birthdate was incomplete (only the year given) or missing then the first date found on the profile was used, or the date was missing altogether. I eliminated the 129 images that had metadata with missing or incomplete dates of birth.

### Face Scores

Each image's corresponding row of metadata contains a primary face score (*face_score*) and a secondary face score (*second_face_score*). The faces in the dataset were detected and scored with a DPM.

![Face Score Distribution](/image_data/Images_for_ReadMe/Primary_Face_Score.jpg)

The face scores in this dataset range from negative infinity (no face was detected and the entire image is included) to approximately 8 (a high probability of a face being detected). An obscured or turned face will lower the score. If multiple faces are detected then the primary face is assigned to the face with the highest score. This creates a problem in group photos, as the IMDB/Wikipedia profile often includes images where the actual person tagged has a lower face score than someone else in the photo.

![A Group Photo With Similar Face Scores](/image_data/Images_for_ReadMe/Two_Separate_Files_With_the_Same_Face_Score.jpg)

The both Geoffrey Arend and Christina Hendricks' IMDB profiles included the above image. Both profiles listed Geoffrey's face as the primary face because it obtained the highest face score. To resolve this issue I created a maximum threshold of secondary face score as a percentage of primary face score to reduce the impact of group photos. For instance, the above picture has a value of 0.8760 (4.0074/4.5745). If the maximum threshold is below this value then this picture would be eliminated from the data set. I eliminated all photos with a primary face score below 1 and photos with a secondary face score as a percentage of primary face score above 0.25. This threshold was selected after trial and error with both celebrity images passed into the system and user images. Of the 511,817 images 214,617 remained.


# The Model

# The Program
