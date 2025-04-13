import cv2
import numpy as np
import matplotlib.pyplot as plt

h,w = 640, 480

bg = cv2.imread("screen.jpg")
# get corners
cap_corner= np.float32([(0,0), (h-1,0),(h-1,w-1), (0,w-1)])
img_corner = np.float32([(275, 190), (618, 90), (625, 390), (280, 390)])
print(cap_corner.shape)
#get the transform matrix from cap_corner to img_corner
m= cv2.getPerspectiveTransform(cap_corner, img_corner)
cap = cv2.VideoCapture(1)
res = np.ones((3, h*w)) #numpy array with (3 rows, h*w cloumns)
#Populating Transformation Result Matrix:
for i in range(h):
    for j in range(w):
        res[0][i*w+j] = i
        res[1][i*w+j] = j
        res[2][i*w+j] = 1
#multiply
res = np.matmul(m,res)
while True:
    _, frame = cap.read()
    for i in range(h): # h as rows
        for j in range(w):
            x = int (res[0,w*i+j]/res[2,w*i+j])
            y = int (res[1,w*i+j]/res[2,w*i+j])
            bg[y][x][:] = frame[j][i][:]
    cv2.imshow("t",bg)
    cv2.waitKey(33)
    