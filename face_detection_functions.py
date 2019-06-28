#Import the necessary libraries

import numpy as np
from numpy import asarray
from numpy import expand_dims

from PIL import Image

import matplotlib.pyplot as plt

from mtcnn.mtcnn import MTCNN

from tensorflow.keras.models import load_model

# Load the FaceNet model
facenet_model = load_model('FaceNet_Model/facenet_keras.h5')
facenet_model.load_weights('FaceNet_Model/facenet_keras_weights.h5')

##########################################################

#####The Main Functions For Image Detection And Prediction#####

#Function 1: Takes in a file path and obtains the pixels of an image
#Converts the image to RGB format if Black and White
def obtain_image_pixels(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    return asarray(image)


#Function 2: Plots the image array given
def plot_face(face_data):
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



#Function 4: Takes in the image array and a face location and returns the resized face as an array
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


#Function 5: Takes in the entire image and list of face locations and outputs a
#list of individually resized faces represented as arrays. You can give the
#model's required image dimensions and a margin.
def get_all_resized_faces(all_image_pixels, dimensions = (160,160), margin = 0):

    all_image_faces = get_all_faces(all_image_pixels)

    face_array_list = []

    #The reason I have the option to return the entire image here is because
    #the imdb/wikipedia images are already cropped to one face. I use the given
    #face score to make decisions about which images to keep. The else statement
    #is primarily for the user images which may contain many faces
    if len(all_image_faces) == 0:
        print('No faces were found, the entire image will process.')
        face_array_list.append(all_image_pixels)
    else:
        for i in all_image_faces:
            face_array_list.append(resize_picture(all_image_pixels, i, dimensions, margin))

    return face_array_list


#Function 6: Takes in one face array and makes a prediction
#This function is adapted from "Deep Learning for Computer Vision"
#Page 488 by Jason Brownlee
def make_a_prediction(face_array):
    face_array = face_array.astype('float32')

    # standardize pixel values across channels
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / std


    # transform face into one observation to be vectorized
    observation = expand_dims(face_array, axis=0)

    yhat = facenet_model.predict(observation)
    return yhat[0]


#Cosine Similarity from Danushka
def findCosineSimilarity(source_representation, user_representation):
    try:
        cos = np.dot(source_representation, user_representation) / (np.sqrt(np.dot(source_representation, source_representation)) * np.sqrt(np.dot(user_representation, user_representation)))
        return cos
    except:
        return 10 #assign a large value in exception. similar faces will have small value.
