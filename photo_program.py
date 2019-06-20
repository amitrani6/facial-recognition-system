#This code is adapted from user derricw from Stack Overflow

import cv2


def take_a_user_picture(file_name):
    cam = cv2.VideoCapture(0)
    cv2.startWindowThread()
    cv2.namedWindow("Input Photo")
    img_counter = 0

    while True:
#         ret = cam.set(3,320)
        ret, frame = cam.read()
        cv2.imshow("Input Photo", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "image_data/user_images/{}".format(file_name)
            cv2.imwrite(img_name, frame)
            img_counter += 1
            break
    cam.release()
    cv2.destroyAllWindows()