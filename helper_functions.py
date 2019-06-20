# -*- coding: utf-8 -*-
"""
Spyder Editor

helper_functions.py

"""

#####Load necessary libraries and external files#####

import pandas as pd
import numpy as np
from numpy import asarray

from PIL import Image

from numpy import expand_dims
from numpy import savez_compressed

from tensorflow.keras.models import load_model

import ast

import os
from os import path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from mtcnn.mtcnn import MTCNN

import cv2
from cv2 import CascadeClassifier
classifier = CascadeClassifier('cascade_models/haarcascade_frontalface_default.xml')

# Load the FaceNet model
facenet_model = load_model('FaceNet_Model/facenet_keras.h5')
facenet_model.load_weights('FaceNet_Model/facenet_keras_weights.h5')

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

print('Loading the dataframe, one moment please.\n')
photo_df.embeddings_fn = photo_df.embeddings_fn.apply(lambda x: convert_csv_to_embeddings(x))
print('The dataframe is now loaded.\n')

##########################################################

#####The Main Functions For Image Detection And Prediction#####

#Function 1: Takes in a file path and obtains the pixels of an image
#Converts the image to RGB format if Black and White
def obtain_image_pixels(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    return asarray(image)

#Function 2: Plots the Face given and array of images
def plot_face(face_data, is_array = True):
    
    #A use case if the array is sent
    if is_array:
        # plot face
        plt.axis('off')
        plt.imshow(face_data)
        plt.show()

#Function 3: Given an image array this function returns the faces detected as a list
#The list is empty if nothing is detected
def get_all_faces(image_array):
    
    #Crop the face further with MTCNN
    detector = MTCNN()
    
    #Obtain the first detected face in the cropped face picture
    faces_detected = detector.detect_faces(image_array)
    
    #MTCNN already returns the list by sorted confidence level
    return faces_detected

#Function 4: Takes in the image array and a face location and returns the resized face as an arrays

def resize_picture(image_array, face_box, dimensions = (160,160), margin = 0):
            
    #Set a margin boolean and while loop to try margin value
    margin_error = True
    
    while margin_error:
    
        try:
        
            # get coordinates
            x1, y1, width, height = face_box['box']
            x2, y2 = x1 + width + margin, y1 + height + margin       
            x1 -= margin
            y1 -= margin 
        
            face_array = image_array[y1:y2, x1:x2]
    
            face_array_resized = Image.fromarray(face_array)
            face_array_resized = face_array_resized.resize(dimensions)
            
            margin_error = False
            break
            
        except:
            
            if margin > 0: 
                margin -= 1
            else:
                face_array_resized = Image.fromarray(image_array)
                face_array_resized = face_array_resized.resize(dimensions)
                break
    
    return asarray(face_array_resized)

#Function 5: Takes in the entire inage and list of face locations and outputs a list of individually resized faces represented as arrays. You can give the model's required image dimensions and a margin.


def get_all_resized_faces(all_image_pixels, dimensions = (160,160), margin = 0):
    
    all_image_faces = get_all_faces(all_image_pixels)
    
    face_array_list = []
    
    #The reason I return the entire image is because the main dataset already lists minimum
    #detected confidence level by face, I remove certain images by this value later
    if len(all_image_faces) == 0:
        print('No faces were found, the entire image will process.')
        face_array_list.append(all_image_pixels)
    else:
        for i in all_image_faces:
            face_array_list.append(resize_picture(all_image_pixels, i, dimensions, margin))
            
    return face_array_list


#Function 6: Takes in one face array and makes a prediction

#This function is adapted from "Deep Learning for Computer Vision" Page 488 by Jason Brownlee

def make_a_prediction(face_array):
    face_array = face_array.astype('float32')
    
    # standardize pixel values across channels
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / std
    
    
    # transform face into one observation to be vectorized
    observation = expand_dims(face_array, axis=0)
    
    yhat = facenet_model.predict(observation)
    return yhat[0]


take_another_picture = True

while take_another_picture:
    
    user_file = input('What is the file name of the picture that you would like to detect? ')
    
    file_exists = False
    
    while not file_exists:
       
        if path.exists("image_data/user_images/" + user_file):
            file_exists = True
            print('Obtaining the image pixels\n\n')
            break
        user_file = input("The file wasn't found, please enter another file: ")
        
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


    #Cosine Similarity from Danushka
    def findCosineSimilarity(source_representation, user_representation):
        try:
            cos = np.dot(source_representation, user_representation) / (np.sqrt(np.dot(source_representation, source_representation)) * np.sqrt(np.dot(user_representation, user_representation)))
            return cos
        except:
            return 10 #assign a large value in exception. similar faces will have small value.

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
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
    a = input("Would you like to take another picture? (Y/N)")
    
    a = a.upper()
    
    if a == 'Y':
        take_another_picture = True
        
    else:
        take_another_picture = False
        
        break
