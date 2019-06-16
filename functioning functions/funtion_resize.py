def resize_picture(filename, dimensions = (160,160), margin = 0):
            
    # load the image
    image = Image.open(filename)
    image = image.convert('RGB')
    image_array = asarray(image)
    
    
    #Set a margin boolean and while loop to try margin value
    margin_error = True
    
    while margin_error:
    
        try:
            #Crop the face further with MTCNN
            detector = MTCNN()
    
            #Obtain the first detected face in the cropped face picture
            first_detected_face = detector.detect_faces(image_array)[0]
        
            # get coordinates
            x1, y1, width, height = first_detected_face['box']
            x2, y2 = x1 + width + margin, y1 + height + margin       
            x1 -= margin
            y1 -= margin 
        
            face_array = image_array[y1:y2, x1:x2]
    
        except:
        
            face_array = image_array
        
        try:
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