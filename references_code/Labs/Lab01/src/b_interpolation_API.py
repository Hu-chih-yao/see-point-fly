import cv2

# 1. read image & error checking
image = cv2.imread('test.jpg')
if image is None:
    print("Error: Loading image failed.")
    exit()

# 2.resize using openCV API
scale=3
output_dimensions = (image.shape[1]*scale, image.shape[0]*scale)
# second element of cv2.resize is (width, height) 
# which is different from imread(heights, widths, channels)
output_image=cv2.resize(image, output_dimensions, interpolation = cv2.INTER_LINEAR)
# 3.save the image
cv2.imwrite ('test_b_interpolation_API.jpg',output_image)