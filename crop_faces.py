import pandas as pd
import numpy as np
from numpy import asarray

from PIL import Image

from numpy import expand_dims

from tensorflow.keras.models import load_model

from mtcnn.mtcnn import MTCNN

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Load the FaceNet model
facenet_model = load_model('FaceNet_Model/facenet_keras.h5')
facenet_model.load_weights('FaceNet_Model/facenet_keras_weights.h5')


#Function 1

def plot_face(face_data, is_array = True):
    
    #A use case if the array is sent
    if is_array:
        # plot face
        plt.axis('off')
        plt.imshow(face_data)
        plt.show()

        
#Obtains the pixel location of all detected faces in the image with MTCNN
def get_all_faces(image_array):
    
    #Crop the face further with MTCNN
    detector = MTCNN()
    
    #Obtain the first detected face in the cropped face picture
    faces_detected = detector.detect_faces(image_array)
    
    #MTCNN already returns the list by sorted confidence level
    return faces_detected


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
    
