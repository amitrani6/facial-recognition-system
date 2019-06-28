# -*- coding: utf-8 -*-
"""

doppelganger_finder.py

"""

print("\n\nOpening The Doppelganger Finder Application.\n\n")

#####Load necessary libraries and external files#####

import pandas as pd
import numpy as np
from numpy import asarray

from PIL import Image

from numpy import expand_dims

import ast

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from mtcnn.mtcnn import MTCNN

from photo_program import *
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

#print(photo_df.head())

##########################################################

#####Convert the dataframes' embedding columns to arrays#####

def convert_csv_to_embeddings(embedding_string):

    #I replace the '\n' and spaces in descending sequential order
    embedding_string = embedding_string.replace('\n', '').replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace('[ ', '[').replace(' ]', ']').replace(' ', ', ')

    #This returns the string as an array in the proper type
    return asarray(ast.literal_eval(embedding_string)).astype('float32')

print('Loading the dataframe, possibly more than one moment please.\n')
photo_df.embeddings_fn = photo_df.embeddings_fn.apply(lambda x: convert_csv_to_embeddings(x))
print('The dataframe is now loaded.\n')

##########################################################



take_another_picture = True

while take_another_picture:

    user_file = 'user.jpg'

    take_a_user_picture(user_file)


    #open camera_take_picture

    image_pixels = obtain_image_pixels("image_data/user_images/{}".format(user_file))
    print('Plotting the image pixels\n\n')
    plot_face(image_pixels)
    print('Obtaining the face areas\n\n')
    face_arrays = get_all_faces(image_pixels)
    print('Obtaining the resized faces\n\n')
    resized_face_arrays = get_all_resized_faces(image_pixels, margin = 20)
    print('Plotting the resized faces\n\n')
#     for i in resized_face_arrays:
#         plot_face(i)

    print('Making Face predictions with FaceNet.\n\n')
    user_embeddings = []

    for i in resized_face_arrays:
        user_embeddings.append(make_a_prediction(i))

    print('Entering the embedding for loop.\n\n')
    for i in range(0,len(user_embeddings)):


        print('Adding Face predictions to dataframe.\n\n')

        photo_df['cosine'] = photo_df['embeddings_fn'].apply(lambda x: findCosineSimilarity(x, user_embeddings[i]))

        print('Plotting the resized image.\n\n')

        plot_face(resized_face_arrays[i])

        print('Sorting predictions with FaceNet.\n\n')

        photo_df = photo_df.sort_values(by = ['cosine'], ascending = False)

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

        if len(user_embeddings) > 1:
            continue_making_predictions = input("Would you like me to make more predictions based on this picture? (Y/N) ")

            continue_making_predictions = continue_making_predictions.lower()

            if "n" in continue_making_predictions:
                break



    a = input("Would you like to take another picture? (Y/N) ")

    a = a.upper()

    if 'Y' in a:
        take_another_picture = True

    else:
        take_another_picture = False

        break
