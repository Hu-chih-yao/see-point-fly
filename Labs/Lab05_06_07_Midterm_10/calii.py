import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

drone = Tello()
drone.connect()
    #time.sleep(10)
drone.streamon()
frame_read = drone.get_frame_read()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
objp=np.zeros((9*6,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
objectPoints=[]
imagePoints=[]
while(len(imagePoints)<=50):
    frame = frame_read.frame
    h,w=frame.shape[:2]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame",frame)
    cv2.waitKey(33)
    ret, corner = cv2.findChessboardCorners(gray, (9,6))
    if ret:
        corner2=cv2.cornerSubPix(gray, corner, (11,11), (-1,-1), criteria)
        objectPoints.append(objp.copy())
        imagePoints.append(corner2)
        frame=cv2.drawChessboardCorners(frame,(9,6),corner2,ret)
    cv2.imshow("frame",frame)
    cv2.waitKey(33)

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (h,w), None,None)
f=cv2.FileStorage("cali.xml", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", cameraMatrix)
f.write("distortion", distCoeffs)
f.release()