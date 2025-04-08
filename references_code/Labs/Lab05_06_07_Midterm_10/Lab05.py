import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID

def main():
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    fs=cv2.FileStorage("test.xml",cv2.FILE_STORAGE_READ)
    intrinsic=fs.getNode("intrinsic").mat()
    distortion=fs.getNode('distortion').mat()
    fs.release()

    while True:
        frame = frame_read.frame
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        # Draw
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        #pose estimation
        if markerCorners!=[]:
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            for i in range(rvec.shape[0]):
                frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[i,:,:], tvec[i,:,:], 10)
                cv2.putText(frame,"x:"+str(tvec[0,0,0])+",y:"+str(tvec[0,0,1])+",z:"+str(tvec[0,0,2]),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        
        cv2.imshow("drone", frame)
        key = cv2.waitKey(33)
    
    #cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

