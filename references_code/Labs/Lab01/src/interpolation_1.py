import cv2
import numpy as np

#1. read the image & check
image = cv2.imread('nctu_flag.jpg')

if image is None:
    print('Error: Could not read the image')
    exit()
#2. set factors & variables
scale =3
#input
#[start:stop:step] from start iterate to stop(exclude) in step width
#image.shape(height,width,channel)
ih,iw =image.shape[:2]
#output
oh,ow = ih*scale, iw*scale
#3. create an black image with desired dimension
output_image = np.zeros((oh,ow,3), dtype=np.uint8) #uint8 is unsigned integer
#4. iterpolation through all pixels of new image
for y in range(oh):# [0] y for height
    for x in range(ow):# [1] w for width
        #take % of the input image into output image
        src_y, src_x = int(y/scale), int(x/scale)
        output_image[y,x] = image[src_y, src_x]
#5. save the new image
cv2.imwrite('interpolation_1.jpg',output_image)

