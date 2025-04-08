import numpy as np
import cv2
cap = cv2.VideoCapture(0)
pic=[]
# Step 1: Preparation of Object Points and Image Points
# Define the dimensions of the chessboard (assuming 9x6 as shown in the image)
objp = np.zeros((9*6, 3), np.float32)
 # 3D points in real world space
 objp[::2]=np.mgrid[0:9, 0:6].t.reshape(-1, 2)
 # 2D points in image plane

# Prepare the object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,5,0)



# Step 2: Read images and detect the chessboard corners


    # Find the chessboard corners

        # Draw and display the corners

# Step 3: Camera Calibration

# Step 4: Save the results
