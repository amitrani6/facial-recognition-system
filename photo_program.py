#This function is adapted from user derricw's answer on from Stack Overflow
#https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv/34588758#34588758

#The function takes in a file name and then takes a picture of the user while
#displaying a camera. Once the photo is taken (by pressing the space bar) the
#photo is saved as the file name given as a parameter of the function. The
#escape key closes the camera window. If the escape key is pressed before the
#picture is taken then no new file is created.

import cv2

def take_a_user_picture(file_name):
    
    #This line opens the primary camera "0" on the device to take a picture
    cam = cv2.VideoCapture(0)
    cv2.startWindowThread()
    cv2.namedWindow("Input Photo")
    img_counter = 0

    while True:
        
        #Each photo taken with the .read() function returns a threshold value
        #'retval' or 'ret' and the frame itself. The threshold value here is
        #the optimal value of pixel intensity that determines if the pixel
        #intensity will be itself or a set value (i.e. 0 {black on grayscale}
        #to 255 {white on grayscale}).
        ret, frame = cam.read()
        
        #Displays the webcam feed on the screen
        cv2.imshow("Input Photo", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            #K in multiples of 27 means the escape key was pressed
            break
        elif k%256 == 32:
            # SPACE pressed
            #K in multiples of 32 means the space bar was pressed
            img_name = "image_data/user_images/{}".format(file_name)
            cv2.imwrite(img_name, frame)
            img_counter += 1
            break
    #Closes the camera and window
    cam.release()
    cv2.destroyAllWindows()