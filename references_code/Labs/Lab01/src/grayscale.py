import numpy as np
import cv2

#1.read image
#we store image into a numpy array of [ rows*columns(pixels) x channels(3) ]
image = cv2.imread('test.jpg')
if image is None:
    print("Error: Could not read the image.")
    exit()
#2.iterate through all rows & columns
for y in range(image.shape[0]): # y for row(height) 
    for x in range(image.shape[1]):# x for column(width)
        #3.store in B->G->R sequence
        (B, G, R) = image[y,x]
        #4.check whether the pixel meet the requirement
        if B>100 and B*0.6>G and B*0.6>R :
            pass
        #5.if not, modify them to gray scale
        else:
            #error fixed, we need to add "int" before RGB, because reading
            #using imread store the data using data type "unit8"(0~255)
            #and the value will exceed it, and wrap around.
            gray_value = int ((int(R)+int(G)+int(B))/3)
            image[y, x] = (gray_value, gray_value, gray_value)
#6.store the image to new file
#save image as new_flag.jpg
cv2.imwrite('test_retain_blue.jpg',image)