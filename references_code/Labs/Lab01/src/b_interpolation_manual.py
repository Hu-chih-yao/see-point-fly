import cv2
import numpy as np

# define the function
def bilinear_interpolation(image, scale):
    #1. set factors & create black output image
    #ih,iw = image.shape[0],image.shape[1]
    ih,iw,channel = image.shape
    oh, ow= ih*scale, iw*scale
    output_image = np.zeros((oh,ow,channel), dtype = image.dtype)

    for y in range (oh):
        for x in range(ow):
            #2. set the x1,x2,y1,y2 for the current pixel
            # we retrieve the data from source image
            src_x = x/scale
            src_y = y/scale

            x1,y1 = int(src_x),int(src_y)
            #we take the second pixel by comparing x1+1 to iw-1, 
            # so that we wouldn't exceed the boundary
            x2,y2 = min(x1+1, iw-1), min (y1+1, ih-1)

            #3. formula
            # as (y2-y1) and (x2-x1) are always 1, so we don't have to write it
            r1 = ((x2-src_x)*image[y1,x1,:] + (src_x-x1)*image[y1,x2,:])
            r2 = ((x2-src_x)*image[y2,x1,:] + (src_x-x1)*image[y2,x2,:])
            output_image[y,x,:] = ((y2-src_y)*r1 + (src_y-y1)*r2).astype(image.dtype)
    return output_image

# read the image & error checking
image = cv2.imread('test.jpg')
if image is None:
    print("Error: Loading image failed.")
    exit()
output_image = bilinear_interpolation(image, 3)
cv2.imwrite('test_b_interpolation_manual.jpg',output_image)