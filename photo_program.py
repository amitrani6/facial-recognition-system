#This function is adapted from user derricw's answer on from Stack Overflow

#The function takes in a file name and then takes a picture of the user while
#displaying a camera. Once the photo is taken (by pressing the space bar) the
#photo is saved as the file name given as a parameter of the function. The
#escape key closes the camera window. If the escape key is pressed before the
#picture is taken then no new file is created.

import cv2

def take_a_user_picture(file_name):
    cam = cv2.VideoCapture(0)
    cv2.startWindowThread()
    cv2.namedWindow("Input Photo")
    img_counter = 0

    while True:
        ret, frame = cam.read()
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

    cam.release()
    cv2.destroyAllWindows()
