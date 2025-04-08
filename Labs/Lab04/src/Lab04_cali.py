import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

# 1 Camera Calibration
drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objectPoints = []
imagePoints = []
while (True):
    frame = frame_read.frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corner = cv2.findChessboardCorners(gray, (6, 9), None)
    if ret:
        objectPoints.append(objp)

        corner2 = cv2.cornerSubPix(gray, corner, (11, 11), (-1, -1), criteria)
        imagePoints.append(corner2)
        cv2.drawChessboardCorners(frame, (6, 9), corner2, ret)
    cv2.imshow("frame", frame)
    cv2.waitKey(1000)
    if len(imagePoints) > 20:
        break

ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints, imagePoints, (frame.shape[1], frame.shape[0]), None, None)
f = cv2.FileStorage("test.xml", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", cameraMatrix)
f.write("distortion", distCoeffs)
f.release()