import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
fb_speed = 40
lf_speed = 40
ud_speed = 50
degree = 30
eps=10
def f1(drone):
    drone.send_rc_control(0, 0, ud_speed, 0)
    time.sleep(1.7)
    drone.send_rc_control(0,fb_speed,0,0)
    time.sleep(1.7)
    drone.send_rc_control(0,0,0,0)
    time.sleep(0.5)
    drone.send_rc_control(0,0,-1*ud_speed,0)
    time.sleep(2.6)
    drone.send_rc_control(0,0,0,0)
def f2(drone):
    drone.send_rc_control(0, 0, -1*ud_speed, 0)
    time.sleep(1)
    drone.send_rc_control(0,fb_speed,0,0)
    time.sleep(5)
    drone.send_rc_control(0,0,0,0)
    drone.land()

    
def clamp(x, max_speed_threshold = 60):
    if x > max_speed_threshold:
        x = max_speed_threshold
    elif x < -max_speed_threshold:
        x = -max_speed_threshold
    return x
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

    x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
    yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    yaw_pid.initialize()

    drone.takeoff()
    #time.sleep(4)
    drone.send_rc_control(0, 0, 50, 0)
    time.sleep(1.8)
    drone.send_rc_control(0,0,0,0)

    while True:
        key = cv2.waitKey(1)
        if key != -1:
            keyboard(drone, key)

        frame = frame_read.frame
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imshow("",frame)
        cv2.waitKey(1)
        h,w=frame.shape[:2]
            
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        
        if markerCorners != []:
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            for i in range(rvec.shape[0]):
                id=markerIds
                if id==0:
                    drone.land()
                if id==1:
                    rotM=np.zeros(shape=(3,3))          
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    yaw = cv2.RQDecomp3x3(rotM)[0]
                    yaw_update = yaw[1] * 1.2
                    x_update = tvec[0, 0, 0] - 10
                    y_update = -(tvec[0, 0, 1] - (-20))
                    z_update = tvec[0, 0, 2] - 100
                    x_update = clamp(x_pid.update(x_update, sleep=0))
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))
                    print(x_update, y_update, z_update, yaw_update)
                    drone.send_rc_control(int(x_update), int(z_update // 2), int(y_update), 0)

                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x:"+str(tvec[0,0,0]-10)+",y:"+str(tvec[0,0,1]+20)+",z:"+str(tvec[0,0,2]-100),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    #cv2.putText(frame,"z:"+str(tvec[0,0,2]),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    if abs(tvec[0,0,0]-10)<eps and abs(tvec[0,0,1]+20)<eps and abs(tvec[0,0,2]-100)<eps:
                        f1(drone)
                if id==2:
                    rotM=np.zeros(shape=(3,3))          
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    yaw = cv2.RQDecomp3x3(rotM)[0]
                    yaw_update = yaw[1] * 1.2
                    x_update = tvec[0, 0, 0] - 10
                    y_update = -(tvec[0, 0, 1] - (-20))
                    z_update = tvec[0, 0, 2] - 100
                    x_update = clamp(x_pid.update(x_update, sleep=0))
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))
                    print(x_update, y_update, z_update, yaw_update)
                    drone.send_rc_control(int(x_update), int(z_update // 2), int(y_update), 0)

                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x:"+str(tvec[0,0,0]-10)+",y:"+str(tvec[0,0,1]+20)+",z:"+str(tvec[0,0,2]-100),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    #cv2.putText(frame,"z:"+str(tvec[0,0,2]),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    if abs(tvec[0,0,0]-10)<eps and abs(tvec[0,0,1]+20)<eps and abs(tvec[0,0,2]-100)<eps:
                        f2(drone)

                cv2.imshow("",frame)
                frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                cv2.waitKey(1)
        else:
            drone.send_rc_control(0,0,0,0)

if __name__ == '__main__':
    main()
