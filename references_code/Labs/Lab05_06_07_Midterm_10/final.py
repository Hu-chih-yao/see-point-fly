import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
fb_speed = 20
lf_speed = 20
up_speed = 20
eps=12.5
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

def clamp(x, max_speed_threshold = 50):
    if x > max_speed_threshold:
        x = max_speed_threshold
    elif x < -max_speed_threshold:
        x = -max_speed_threshold
    return x

'''initialize'''
drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
x_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
fs=cv2.FileStorage("cali.xml",cv2.FILE_STORAGE_READ)
intrinsic=fs.getNode("intrinsic").mat()
distortion=fs.getNode('distortion').mat()
fs.release()
x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()
'''takeoff and up'''
drone.takeoff()
drone.send_rc_control(0,0,0,0)
while True:
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)

    frame = frame_read.frame
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    h,w=frame.shape[:2]

    cv2.imshow("",frame)
        
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    
    if s2 or s3 or s4 or s5 or s6 or s7:
        '''trace black'''
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)    #gray
        ret,frame=cv2.threshold(frame, 40, 255, cv2.THRESH_BINARY)
        frame=cv2.GaussianBlur(frame,(5,5),0)

        if s3 or s5 or s7:
            tmp=0
            for i in range (10):
                for j in range(40):
                    if s3:
                        tmp+=frame[h//4+i,(w//2)-30+j]
                    if s5 or s7:
                        tmp+=frame[(3*h)//4+i,(w//2)-30+j]
            # print(tmp)

        if (s2 or s4) and (sum(frame[h//2-30][0:10])!=0):
            drone.send_rc_control(0,0,up_speed,0)
        elif s3 and (tmp!=0):
            drone.send_rc_control(-1*lf_speed,0,0,0)
        elif (s5 or s7) and (tmp!=0):
            drone.send_rc_control(-1*lf_speed,0,0,0)
        elif s6 and (sum(frame[h//2-30][0:10])!=0):
            drone.send_rc_control(0,0,-1*up_speed,0)
        elif s7:
            drone.land()
        else:
            drone.send_rc_control(0, 0, 0, 0)
            if s2: s2=0; s3=1; print("start s3"); 
            elif s3: s3=0; s4=1; print("start s4"); drone.move("up",20)
            elif s4: s4=0; s5=1; print("start s5"); drone.move("left",20)
            elif s5: s5=0; s6=1; print("start s6")
            elif s6: s6=0; s7=1; print("start s7"); drone.move("left",20)
    if s1 and markerCorners != []:
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
        for i in range(rvec.shape[0]):
            id=markerIds[i]
            if id==1:
                z1=75
                rotM=np.zeros(shape=(3,3))          
                cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                yaw = cv2.RQDecomp3x3(rotM)[0]
                yaw_update = yaw[1] * 1.2
                x_update = tvec[i,0,0] - xi
                y_update = -(tvec[i,0,1] - yi)
                z_update = tvec[i,0,2] - z1
                x_update = clamp(x_pid.update(x_update, sleep=0))
                y_update = clamp(y_pid.update(y_update, sleep=0))
                z_update = clamp(z_pid.update(z_update, sleep=0))
                yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))
                drone.send_rc_control(int(x_update), int(z_update *0.75), int(y_update), int(yaw_update//2))
                frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                cv2.putText(frame,"x: "+str(round(tvec[i,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame,"y: "+str(round(tvec[i,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                cv2.putText(frame,"z: "+str(round(tvec[i,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                if abs(tvec[i,0,0]-xi)<10 and abs(tvec[i,0,1]-yi)<eps and abs(tvec[i,0,2]-z1)<eps:
                    s1=0; s2=1; print("start s2") 
    else:
        drone.send_rc_control(0,0,0,0)
    cv2.imshow("",frame)
    cv2.waitKey(1)