# facial-recognition-system

The **DOPPELGÄNGER: Facial Recognition System** is a program that determines the celebrity or politician that a person most closely resembles using the FaceNet CNN and images from IMDB and Wikipedia.

## Repository Contents

The following 12 files and 3 folders are included in this repository:

  1.	**FaceNet_Model/** - A folder containing the Keras FaceNet model and weights
  2.	**Photo_Dataframes/** - A folder containing the metadata and vector embeddings of all faces in the database
  3.	**image_data/** - A folder containing the user images and imdb/wikipedia images (for size purposes a download link is included)
  4.	**.gitignore** - Contains the files excluded from this repository (images that collectively take up too much space and test notebooks)
  5.	**EDA.ipynb** - Contains both Exploratory Data Analysis and the methodology for eliminating irrelevant images 
  6.	**FaceNet Obtain Embeddings.ipynb** - A notebook for obtaining the facial embeddings vector from the FaceNet model
  7.	**README.md** - This document
  8.	**Keras_FaceNet_Model_Structure.ipynb** - A notebook that creates a summary chart of the Keras FaceNet model's structure
  9.	**doppelganger_finder.py** - The main file; run this document with the requirements file listed to use the program
  10.	**environment.yml** - A file for replicating the environment required to run **doppelganger_finder.py**
  11.	**face_detection_functions.py** - A python script containing seven functions for the detection, extraction and comparison of faces using the FaceNet model and MTCNN library 
  12.	**imdb_metadata_cleanup.ipynb** - The metadata for the images from IMDB (conversion from matlab)
  13.	**photo_program.py** - A python file containing the function to take a picture with a device's camera using OpenCV
  14.	**requirements.txt** - The requirements file for running **doppelganger_finder.py**
  15.	**wiki_metadata_cleanup.ipynb** - The metadata for the images from Wikipedia (conversion from matlab)

# The Data

The images and metadata come from ETH Zürich's Computer Vision Lab. There are 460,723 images from IMDB and 62,328 images from Wikipedia that correspond to +70k individuals. The data was collected in 2015 by Rasmus Rothe, Radu Timofte, and Luc Van Gool for their papers on age detection ([Deep expectation of real and apparent age from a single image without facial landmarks](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)). The faces are already cropped by the ETH Zürich team using a Deformable Parts Model (DPM) described in the paper [Face Detection Without The Bells And Whistles](http://rodrigob.github.io/documents/2014_eccv_face_detection_with_supplementary_material.pdf) (*by Markus Mathias, Rodrigo Benenson, Marco Pedersoli, and Luc Van Gool*). I downloaded the two cropped faces .tar files which total 8 GB.

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

The **EDA.ipynb** notebook contains both the Exploratory Data Analysis and the methodology for eliminating images that are either missing metadata or have low face scores. There are 523,051 images in the initial dataset corresponding to +70k individuals.

### Missing Names

I eliminated the 124 images with missing names as I would not be able to associate a user's face to them correctly.

### Birthday Conversion

![Age Distribution](/image_data/Images_for_ReadMe/Age.jpg)

The birthdays associated with each profile and image were webscraped off of IMDB and Wikipedia. If the birthdate was incomplete (only the year given) or missing then the first date found on the profile was used, or the date was missing altogether. I eliminated the 129 images that had metadata with missing or incomplete dates of birth.

### Face Scores

Each image's corresponding row of metadata contains a primary face score (*face_score*) and a secondary face score (*second_face_score*). The faces in the dataset were detected and scored with the DPM from the above paper.

![Face Score Distribution](/image_data/Images_for_ReadMe/Primary_Face_Score.jpg)

The face scores in this dataset range from negative infinity (no face was detected and the entire image is included) to approximately 8 (a high probability of a face being detected). An obscured or turned face will have a lower score. If multiple faces are detected then the primary face is assigned to the face with the highest score. This creates a problem with group photos, as the IMDB/Wikipedia profile often includes images where the actual person tagged has a lower face score than someone else in the photo.

![A Group Photo With Similar Face Scores](/image_data/Images_for_ReadMe/Two_Separate_Files_With_the_Same_Face_Score.jpg)

Both Geoffrey Arend's and Christina Hendricks' IMDB profiles included the above image. Both profiles listed Geoffrey's face as the primary face because it obtained the highest face score. To resolve this issue I created a maximum threshold of secondary face score as a percentage of primary face score to reduce the impact of group photos. For instance, the above picture has a value of 0.8760 (4.0074/4.5745). If the maximum threshold is below this value then this picture would be eliminated from the data set. I eliminated all photos with a primary face score below 1 and photos with a secondary face score as a percentage of primary face score above 0.25. This threshold was selected after trial and error with both celebrity images passed into the system and user images. Of the 511,817 images 214,617 remained belonging to 48,138 individuals.

*In the **EDA.ipynb** notebook I concatenate the two .csv files from IMDB and Wikipedia into one dataframe and save it as **All_Photo_Data.csv** in the **Photo_Dataframes/** folder.*

# The Model

I used a Keras implementation of the FaceNet Convolutional Neural Network (CNN) model to obtain face embeddings for each image in the dataset and for each face in the user image. Face Embeddings are vector representations of a face which can be compared to each other with a similarity metric; in FaceNet's case these vectors have a length of 128 elements.

The process of obtaining a face embedding from FaceNet is:

  1. **Crop the faces from an image** Because my dataset of images is already cropped to the most prominent face, I can skip the initial cropping of the faces. I will later describe how I crop faces when I discuss user images below.
  2. **Input a resized face image.** The FaceNet model requires all input image tensors to be the size 160x160x3. This means that the image width and length is to be 160 pixels by 160 pixels and that there must be three color channels in the RGB format. If the images are grayscale then they must be converted.
  3. **Normalize the pixels** Once the image is resized the pixels must be normalized using z-score normalization, as required by the FaceNet model. In z-score normalization the mean of all the pixels is subtracted from each pixel and then divided by the standard deviation of the pixels.
  4. **Add a dimension to the image tensor** A fourth dimension is added to the tensor so the model can keep track of what sample/observation group the image belongs to while it moves through each layer of the model.
  5. **Obtain the face embedding vector** The face embeddings for each image is obtained by passing the image tensor through the FaceNet Model with the model.predict() method.
  
 The architecture of the FaceNet model is below:

![FaceNet Architecture](/image_data/Images_for_ReadMe/FaceNet_Layers.png)

Source: [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf) (*by Florian Schroff, Dmitry Kalenichenko, and James Philbin of Google*)

*The above is a summary structure chart of the FaceNet NN1 Inception newtwork architecture which can be found in the original model paper cited above. I used the FaceNet NN3 Inception architecture which takes an input size of 160x160x3. The complete model visualization chart was generated in **Keras_FaceNet_Model_Structure.ipynb** and is stored in **image_data/Images_for_ReadMe/** as **FaceNet_Model_Structure.png**. I used Hiroki Taniai's Keras implementation which can be found [here](https://github.com/nyoki-mtl/keras-facenet).*

The FaceNet CNN works by first picking up minor facial features in convolution layers utilizing rectified linear unit (ReLU) activation functions. The existance of features is then combined in pooling layers (*FaceNet uses max pooling to summarize the features of the previous layers*). This step determines if more complex facial patterns exist in a specific observation vector. There are also normalization layers that rescale the results of the activation functions (removing the mean of the vector and dividing by the standard deviation of the vector) before they are passed through to further layers. Each face vector is then concatenated in the final layers and normalized one final time achieving a final embedding size of 128 elements. Face embeddings from FaceNet belonging to the same person will have a smaller distance than face embeddings belonging to two different people. When used as a classifier, as in the task of facial recognition, natural changes will not affect the identification ability of FaceNet (*i.e. changes in hair styles/color, wearing accessories like glasses, etc.*).

I selected the FaceNet model over the Visual Geometry Group (VGG16) model for two reasons: The first reason is that FaceNet achieved better results on benchmark datasets; The second reason is that the embedding vector produced by FaceNet is only 128 elements long while the vector from the VGG model is 2622 elements long. The VGG vectors would have been too large to store in the .csv format that I use. The VGG architecture can be found here: [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) (*by Omkar M. Parkhi, Andrea Vedaldi, and Andrew Zisserman*).

I obtain the face embeddings for each image in the dataset in the notebook **FaceNet Obtain Embeddings.ipynb**. I split the main dataframes into five dataframes due to the expected increase in size of the .csv files. The embeddings are added to the new dataframes in a newly created column and then saved in the folder **Photo_Dataframesphoto_dfs_with_embeddings_fn/**. Each dataframe is no bigger than 92 MB.

*I originally obtained 372,917 face embeddings which took 8 hours, 57 minutes, and 10 seconds on my local machine. After I tested the program I reduced the number of images in my dataset to 214,617 using the process described in the section: **EDA/Image Selection**.*

# The Program

The file **doppelganger_finder.py** is the facial recognition python script; it calls **photo_program.py** to take a user photo and **face_detection_functions.py** to evaluate the user photo taken.

## photo_program.py

**photo_program.py** contains one function that uses the OpenCV library to (1) open the primary camera on a device, (2) display a camera window on the screen, (3) take a photo with that camera, and (4) save that image as **user.jpg** in the folder **image_data/user_images/**.

## face_detection_functions.py

**face_detection_functions.py** contains seven functions.




doppelganger_finder.py
environment.yml
face_detection_functions.py
photo_program.py
requirements.txt
