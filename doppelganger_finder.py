# -*- coding: utf-8 -*-
"""

doppelganger_finder.py

"""

#doppelganger_finder.py is the main file of the facial recognition system
#This python script will take a picture of one or more users and display
#the most similar faces by cosine similarity

##########################################################################

print("\n\nOpening The Doppelganger Finder Application.\n\n")

#####Load necessary libraries and external files#####

import pandas as pd
import numpy as np
from numpy import asarray

from PIL import Image

from numpy import expand_dims

import ast

#There are several Keras/TensorFlow deprecation warnings about
#future updates, I use the following code to ignore some of them
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from mtcnn.mtcnn import MTCNN

#This is the file with the function that takes a picture with the devices' webcam
from photo_program import *
#This is the file with the functions that locate a face within a picture, use
#the Keras FaceNet CNN model, and faces by cosine similarity
from face_detection_functions import *

import cv2

##########################################################

#####Load The Dataframe with embeddings#####

photo_df_a = pd.read_csv('Photo_Dataframes/photo_dfs_with_embeddings_fn/photo_dataframe_fn_embeddings_a.csv')
photo_df_b = pd.read_csv('Photo_Dataframes/photo_dfs_with_embeddings_fn/photo_dataframe_fn_embeddings_b.csv')
photo_df_c = pd.read_csv('Photo_Dataframes/photo_dfs_with_embeddings_fn/photo_dataframe_fn_embeddings_c.csv')
photo_df_d = pd.read_csv('Photo_Dataframes/photo_dfs_with_embeddings_fn/photo_dataframe_fn_embeddings_d.csv')
photo_df_e = pd.read_csv('Photo_Dataframes/photo_dfs_with_embeddings_fn/photo_dataframe_fn_embeddings_e.csv')

#####Concat the dataframes#####
photo_df = pd.concat([photo_df_a, photo_df_b, photo_df_c, photo_df_d, photo_df_e]).reset_index(drop=True)

##########################################################

#####Convert the dataframes' embedding columns to arrays#####

def convert_csv_to_embeddings(embedding_string):

    #I replace the '\n' and spaces in descending sequential order
    embedding_string = embedding_string.replace('\n', '').replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace('[ ', '[').replace(' ]', ']').replace(' ', ', ')

    #This returns the string as a numpy array
    return asarray(ast.literal_eval(embedding_string)).astype('float32')

print('\nLoading the dataframe, possibly more than one moment please.\n')
photo_df.embeddings_fn = photo_df.embeddings_fn.apply(lambda x: convert_csv_to_embeddings(x))
print('The dataframe is now loaded.\n')

##########################################################

#I use a while loop to run my program until the variable 'take_another_picture'
#is equal to False. The process is as follows:
#       1. Take the user picture
#       2. Obtain the face embeddings vector for the user picture
#       3. Calculate the cosine similarity between the faces in the user picture
#          and the faces in the data frame
#       4. Output the results
#       5. Repeat the process if 'take_another_picture' equals True

take_another_picture = True

while take_another_picture:

    #user_file is the name of the file to be saved
    user_file = 'user.jpg'

    #take_a_user_picture takes a photo of a user and saves the file
    #This function comes from 'photo_program.py'
    take_a_user_picture(user_file)

    #The following functions come from 'face_detection_functions.py' please see
    #this file for additional  comments

    #'image_pixels' is a three dimensional tensor representing the RGB channels
    #of the user image as a numpy array data type
    image_pixels = obtain_image_pixels("image_data/user_images/{}".format(user_file))
    print('Plotting the image pixels\n\n')

    plot_face(image_pixels)
    print('Obtaining the face areas\n\n')

    #'face_arrays' is a list of the locations of all the faces in the image
    face_arrays = get_all_faces(image_pixels)
    print('Obtaining the resized faces\n\n')

    #'resized_face_arrays' is a list of resized faces stored as face_arrays
    #The size of each face is adjusted to 160 by 160 as this is the required size
    #of the FaceNet model. A margin of twenty pixels is added to the face to
    #prevent the loss of valuable face features on the edge of the face
    resized_face_arrays = get_all_resized_faces(image_pixels, margin = 20)
    print('Plotting the resized faces\n\n')

    print('Making Face predictions with FaceNet.\n\n')
    user_embeddings = []

    for i in resized_face_arrays:
        #This line of code obtains the 128 length vector face embeddings
        user_embeddings.append(make_a_prediction(i))

    #This for loop runs through each detected face in the list of face embeddings
    print('Entering the embedding for loop.\n\n')
    for i in range(0,len(user_embeddings)):

        #Adds the cosine similarity of the specific user to all images in the
        #data frame
        print('Adding Face predictions to dataframe.\n\n')
        photo_df['cosine'] = photo_df['embeddings_fn'].apply(lambda x: findCosineSimilarity(x, user_embeddings[i]))

        print('Plotting the resized image.\n\n')
        plot_face(resized_face_arrays[i])

        #Sorts the data frame by cosine similarity in descending order
        print('Sorting predictions with FaceNet.\n\n')
        photo_df = photo_df.sort_values(by = ['cosine'], ascending = False)

        #displays the three most similar faces to the user face
        print('About to print predictions with FaceNet.\n\n')
        for i in range(0, 3):
            instance = photo_df.iloc[i]
            name = instance['name']
            distance = instance['cosine']
            img = cv2.imread(instance['file_path'])
            print(str(i+1),".",name," (",instance['cosine'],")")
            plt.title(instance['name'] + ", cosine similarity: " + str(round(instance['cosine'], 4)))
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

        #I give the option to skip over user faces if multiple faces are detected.
        #This is because I did not want to flip through multiple faces that were
        #detected in the background of the user picture.
        if len(user_embeddings) > 1:
            continue_making_predictions = input("Would you like me to make more predictions based on this picture? (Y/N) ")

            continue_making_predictions = continue_making_predictions.lower()

            if "n" in continue_making_predictions:
                break


    #The following code is for the option to take another user picture.
    a = input("Would you like to take another picture? (Y/N) ")
    a = a.upper()

    if 'Y' in a:
        take_another_picture = True

    else:
        take_another_picture = False

        break
