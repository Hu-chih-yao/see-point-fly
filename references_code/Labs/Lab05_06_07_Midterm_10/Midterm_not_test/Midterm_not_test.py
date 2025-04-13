import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard

fb_speed = 50
lf_speed = 50
ud_speed = 50
degree = 90
s1=1
s2=0
s0=0
s3=0
s4=0
s5=0

def f1(drone):
    drone.send_rc_control(0, 0, ud_speed, 0)
    time.sleep(1.6)   # 1.6 good
    drone.send_rc_control(0,fb_speed,0,0)
    time.sleep(1)     # 1.25 much
    drone.send_rc_control(0,0,0,0)
    time.sleep(0.1)
    drone.send_rc_control(0,0,-1*ud_speed,0)
    time.sleep(3)
    drone.send_rc_control(0,0,0,0)
    # drone.move('up',80)
    # drone.move('forward',50)
    # drone.move('down',150)
    global s1, s2
    s1=0
    s2=1
def f2(drone):
    drone.send_rc_control(0, 0, -1*ud_speed, 0)
    time.sleep(1)
    drone.send_rc_control(0,fb_speed,0,0)
    time.sleep(2.75)   #2.5-3 
    drone.send_rc_control(0, 0, ud_speed, 0)
    time.sleep(2)    #
    drone.send_rc_control(0,0,0,0)
    # drone.move('down',50)
    # drone.move('forward',137.5)
    # drone.move('up',100)
    global s2, s0
    s2=0
    s0=1
def f3(drone):
    drone.move("forward",25)
    drone.rotate_clockwise(90)
    global s0, s3, s4
    s0=0
    s3=0
    s4=1
def f4(drone):
    drone.move("left",220)  #225 jiejin
    global s5, s4
    s5=1
    s4=0
def f5(drone):
    drone.move("back",10)   #
    drone.land()

def clamp(x, max_speed_threshold = 50):
    if x > max_speed_threshold:
        x = max_speed_threshold
    elif x < -max_speed_threshold:
        x = -max_speed_threshold
    return x

'''main'''
drone = Tello()
drone.connect()
drone.streamon()

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)   # Load the predefined dictionary 
parameters = cv2.aruco.DetectorParameters_create()  # Initialize the detector parameters using default values

fs=cv2.FileStorage("test.xml",cv2.FILE_STORAGE_READ)
intrinsic=fs.getNode("intrinsic").mat()
distortion=fs.getNode('distortion').mat()
fs.release()

x_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
y_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
z_pid   = PID(kP=0.7, kI=0.0001, kD=0.1)
yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

x_pid.initialize()
y_pid.initialize()
z_pid.initialize()
yaw_pid.initialize()

xi=10
yi=-20
eps=12.5    #12.5 good

def dw(zi,rvec,tvec,drone,yaw_multi):
    rotM=np.zeros(shape=(3,3))          
    cv2.Rodrigues(rvec, rotM, jacobian=0)
    yaw = cv2.RQDecomp3x3(rotM)[0]
    x_update =   tvec[0, 0, 0]-xi
    y_update = -(tvec[0, 0, 1]-yi)
    z_update =   tvec[0, 0, 2]-zi
    yaw_update = yaw[1]
    x_update = clamp(x_pid.update(x_update, sleep=0))
    y_update = clamp(y_pid.update(y_update, sleep=0))
    z_update = clamp(z_pid.update(z_update, sleep=0))
    yaw_update=clamp(yaw_pid.update(yaw_update, sleep=0))
    drone.send_rc_control(int(x_update), int(z_update/2), int(y_update), int(yaw_update*yaw_multi))

while True:
    key = cv2.waitKey(1)
    if key != -1:
        keyboard(drone, key)

    frame_read = drone.get_frame_read()
    frame = frame_read.frame
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    cv2.imshow("1",frame)
    cv2.waitKey(1)
        
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)    # Detect the markers in the image
    if markerCorners != []:
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)  #Pose estimation for single markers.
        id=markerIds[0]
        i=0
        if s1 and id==1:
            z1=100
            dw(z1,rvec[i],tvec,drone,0.5)
            frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)  # Draw marker, id
            frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[0,:,:],tvec[0,:,:],10)
            cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            if abs(tvec[0,0,0]-xi)<eps and abs(tvec[0,0,1]-yi)<eps and abs(tvec[0,0,2]-z1)<eps:
                # f1(drone)
                drone.move('up',80)
                drone.move('forward',50)
                drone.move('down',150)
                s1=0
                s2=1
        elif s2 and id==2:
            z2=100
            dw(z2,rvec[i],tvec,drone,0.5)
            frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
            frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[0:,:],tvec[0,:,:],10)
            cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            if abs(tvec[0,0,0]-xi)<eps and abs(tvec[0,0,1]-yi)<eps and abs(tvec[0,0,2]-z2)<eps:
                # f2(drone)    
                drone.move('down',50)
                drone.move('forward',137.5)
                drone.move('up',100)
                s2=0
                s0=1
        elif s0 and id==0:
            z0=100
            dw(z0,rvec[i],tvec,drone,2.2)
            frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
            frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[0,:,:],tvec[0,:,:],10)
            cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            s3=1
        elif s3 and id==3:
            z3=100
            dw(z3,rvec[i],tvec,drone,1)
            frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
            frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
            cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            if abs(tvec[0,0,0]-xi)<eps and abs(tvec[0,0,1]-yi)<eps and abs(tvec[0,0,2]-z3)<eps:
                # f3(drone)
                drone.move("forward",25)
                drone.rotate_clockwise(90)
                s0=0
                s3=0
                s4=1
        elif s4 and id==4:
            z4=125
            dw(z4,rvec[i],tvec,drone,0.5)
            frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
            frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
            cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            if abs(tvec[0,0,0]-xi)<eps and abs(tvec[0,0,1]-yi)<eps and abs(tvec[0,0,2]-z4)<eps:
                f4(drone)    
                drone.move("left",220)  #225 jiejin
                s5=1
                s4=0
        elif s5 and id==5:
            z5=125
            dw(z5,rvec[i],tvec,drone,0.5)
            frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
            frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
            cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            if abs(tvec[0,0,0]-xi)<eps and abs(tvec[0,0,1]+yi)<eps and abs(tvec[0,0,2]-z5)<eps:
                # f5(drone)
                drone.move("back",10)   #
                drone.land()
                break
        else:
            frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
            print("many aruco")
        cv2.imshow("2",frame)
        cv2.waitKey(1)
    else:
        drone.send_rc_control(0,-10,0,0)  # Not detect any marker => backward

print("over")
