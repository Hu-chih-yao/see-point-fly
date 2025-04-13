import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
fb_speed = 20
lf_speed = 20
ud_speed = 20
degree = 90
eps=12.5    #12.5 good
xi=5    #10
yi=-10   #-20
s1=1
s2=0
s3=0
s4=0
s5=0
s6=0
s7=0
s8=0
h=0; w=0

def clamp(x, max_speed_threshold = 50):
    if x > max_speed_threshold:
        x = max_speed_threshold
    elif x < -max_speed_threshold:
        x = -max_speed_threshold
    return x

# Tello
drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

# Load the predefined dictionary 
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()

x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

fs=cv2.FileStorage("cali.xml",cv2.FILE_STORAGE_READ)
intrinsic=fs.getNode("intrinsic").mat()
distortion=fs.getNode('distortion').mat()
fs.release()

z_pid.initialize()
y_pid.initialize()
x_pid.initialize()
yaw_pid.initialize()
'''begin takeoff and up'''
drone.takeoff()
drone.send_rc_control(0,0,0,0)
while True:
    
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)

    frame = frame_read.frame
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)    #gray
    ret,gray=cv2.threshold(gray, 31, 255, cv2.THRESH_BINARY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    #frame=cv2.Canny(frame,100,200)

    cv2.imshow("",gray)
    cv2.imshow("",frame)
    h,w=frame.shape[:2]
        
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    
    if s2:
        if(gray[h//4,w//2]!=0):
            drone.send_rc_control(lf_speed , 0, 0, 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            s3=1; s2=0
            print("start s3")
    if s3:
        if(gray[h//2,(3*w)//4]!=0):
            drone.send_rc_control(0, 0, ud_speed, 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            s4=1; s3=0
            print("start s4")
    if s4:
        if(gray[h//4,w//2]!=0):
            drone.send_rc_control(lf_speed, 0, 0, 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            s5=1; s4=0
            drone.move("up",20)
            print("start s5")
    if s5:
        if(gray[h//2,0]!=0):
            drone.send_rc_control(0, 0, ud_speed, 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            s6=1; s5=0
            drone.move("left",20)
            print("start s6")
    if s6:
        if(gray[(13*h)//16,w//2]!=0):
            drone.send_rc_control(-1*lf_speed, 0, 0, 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            s7=1; s6=0
            drone.move("down",20)
            print("start s7")
    if s7:
        if(gray[h//2,w-1]!=0):
            drone.send_rc_control(0, 0, -1*ud_speed, 0)
        else:
            drone.send_rc_control(0, 0, 0, 0)
            drone.move("back",25)
            s8=1; s7=0
            print("start s8")
    if markerCorners != []:
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
        for i in range(rvec.shape[0]):
            id=markerIds[i]
            if s8 and id==4:
                drone.land()
                print("finish")
            if s1 and id==4:
                z4=60
                rotM=np.zeros(shape=(3,3))          
                cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                yaw = cv2.RQDecomp3x3(rotM)[0]
                yaw_update = yaw[1] * 1.2
                x_update = tvec[i,0,0] - xi
                y_update = -(tvec[i,0,1] - yi)
                z_update = tvec[i,0,2] - z4
                x_update = clamp(x_pid.update(x_update, sleep=0))
                y_update = clamp(y_pid.update(y_update, sleep=0))
                z_update = clamp(z_pid.update(z_update, sleep=0))
                yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))
                drone.send_rc_control(int(x_update//2), int(z_update *0.75), int(y_update), 0)
                frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                cv2.putText(frame,"x: "+str(round(tvec[i,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame,"y: "+str(round(tvec[i,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame,"z: "+str(round(tvec[i,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                if abs(tvec[i,0,0]-xi)<eps and abs(tvec[i,0,1]-yi)<eps and abs(tvec[i,0,2]-z4)<eps:
                    s2=1; s1=0
                    print("start s2")
                    drone.move("right",20)
    else:
        drone.send_rc_control(0,0,0,0)
    cv2.imshow("",gray)
    cv2.imshow("",frame)
    cv2.waitKey(1)