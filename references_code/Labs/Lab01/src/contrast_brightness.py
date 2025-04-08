import numpy as np
import cv2

#0. set desired contrast and brightness
contrast = 100
brightness=40
#1. read the image & convert to int32 to avoid overflow
image = cv2.imread('test.jpg')
if image is None:
    print("Error: failed to read the image")
    exit()
image_int32 = np.array(image, dtype=np.int32)

#2. iterate through all rows & columns
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        (B, G, R) = image_int32[y, x]
        
        #3. check whether the pixel is blue or yellow
        if (B + G) * 0.3 > R:
            #4. adjust the contrast & brightness
            #this step modify all the 3 channel of image_int32[y,x]
            new_pixel = (image_int32[y,x]-127)*(contrast/127+1)+127+brightness
            #5. clip the value
            new_pixel = np.clip(new_pixel, 0, 255)
            #update the original int8 Pixel with the new one
            image[y, x] = np.array(new_pixel, dtype=np.int8)

#save the file
cv2.imwrite('test_contrast_brightness.jpg',image)