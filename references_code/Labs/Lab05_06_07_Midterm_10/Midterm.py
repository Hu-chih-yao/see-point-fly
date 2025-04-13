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
eps=12.5    #12.5 good
s1=1
s2=0
s0=0
s3=0
s4=0
s5=0
tko=0
xi=5    #10
yi=-10   #-20

def f1(drone):
    drone.send_rc_control(0, 0, ud_speed, 0)
    time.sleep(1.6)    #1.6 good
    drone.send_rc_control(0,fb_speed,0,0)
    time.sleep(1) #  1 good
    drone.send_rc_control(0,0,0,0)
    time.sleep(0.1)
    drone.send_rc_control(0,0,-1*ud_speed,0)
    time.sleep(3.5)   #3 less
    drone.send_rc_control(0,0,0,0)
    global s2, s1
    s2=1
    s1=0
def f2(drone):
    drone.send_rc_control(0, 0, -1*ud_speed, 0)
    time.sleep(1)
    drone.send_rc_control(0,fb_speed,0,0)
    time.sleep(3)   #2.5-2.75
    drone.send_rc_control(0, 0, ud_speed, 0)
    time.sleep(2)    #
    drone.send_rc_control(0,0,0,0)
    global s2, s0
    s0=1
    s2=0
def f3(drone):
    drone.rotate_clockwise(90)
    global s4, s3,s0
    s4=1
    s3=0
    s0=0
def f4(drone):
    drone.move("left",220)  #225 jiejin
    global s5, s4
    s5=1
    s4=0
def f5(drone):
    drone.move("back",25)   #
    drone.land()
    return 0
def clamp(x, max_speed_threshold = 50):
    if x > max_speed_threshold:
        x = max_speed_threshold
    elif x < -max_speed_threshold:
        x = -max_speed_threshold
    return x
def main():
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

    fs=cv2.FileStorage("test.xml",cv2.FILE_STORAGE_READ)
    intrinsic=fs.getNode("intrinsic").mat()
    distortion=fs.getNode('distortion').mat()
    fs.release()
    
    z_pid.initialize()
    y_pid.initialize()
    x_pid.initialize()
    yaw_pid.initialize()
    '''begin takeoff and up'''
    drone.takeoff()
    time.sleep(1)
    drone.send_rc_control(0, 0, 10, 0)
    time.sleep(3)
    drone.send_rc_control(0,0,0,0)
    global s1, s2, s3, s4, s5, s0
    while True:
        key = cv2.waitKey(1)
        if key != -1:
            keyboard(drone, key)

        frame = frame_read.frame
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imshow("",frame)
        h,w=frame.shape[:2]
            
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        
        if markerCorners != []:
            print(markerIds)
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
            for i in range(rvec.shape[0]):
                id=markerIds[i]
                if s1 and id==1:
                    global tko
                    tko=1
                    rotM=np.zeros(shape=(3,3))          
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    x_update = tvec[i, 0, 0] - 10
                    y_update = -(tvec[i, 0, 1] - (-20))
                    z_update = tvec[i, 0, 2] - 100
                    x_update = clamp(x_pid.update(x_update, sleep=0))
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    drone.send_rc_control(int(x_update), int(z_update // 2), int(y_update), 0)
                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    if abs(tvec[i,0,0]-10)<eps and abs(tvec[i,0,1]+20)<eps and abs(tvec[i,0,2]-100)<eps:
                        f1(drone)
                if s2 and id==2:
                    rotM=np.zeros(shape=(3,3))
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    x_update = tvec[i, 0, 0] - xi
                    y_update = -(tvec[i, 0, 1] - yi)
                    z_update = tvec[i, 0, 2] - 100
                    x_update = clamp(x_pid.update(x_update, sleep=0))
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    drone.send_rc_control(int(x_update), int(z_update // 2), int(y_update), 0)
                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x: "+str(round(tvec[0,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"y: "+str(round(tvec[0,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"z: "+str(round(tvec[0,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    if abs(tvec[i,0,0]-xi)<eps and abs(tvec[i,0,1]-yi)<eps and abs(tvec[i,0,2]-100)<eps:
                        f2(drone)
                if s0 and id==0:
                    z0=75
                    rotM=np.zeros(shape=(3,3))          
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    yaw = cv2.RQDecomp3x3(rotM)[0]
                    yaw_update = yaw[1] * 1.2
                    x_update = tvec[i, 0, 0] - xi
                    y_update = -(tvec[i, 0, 1] - yi)
                    z_update = tvec[i, 0, 2] - z0
                    x_update = clamp(x_pid.update(x_update, sleep=0))
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))
                    drone.send_rc_control(int(x_update//2), int(z_update*0.75), int(y_update), int(yaw_update*2)) #yaw x 2
                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x: "+str(round(tvec[i,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"y: "+str(round(tvec[i,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"z: "+str(round(tvec[i,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    s3=1
                if s3 and id==3:
                    z3=75   # 50 too close
                    rotM=np.zeros(shape=(3,3))          
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    yaw = cv2.RQDecomp3x3(rotM)[0]
                    yaw_update = yaw[1] * 1.2
                    x_update = tvec[i, 0, 0] - xi
                    y_update = -(tvec[i, 0, 1] - yi)
                    z_update = tvec[i, 0, 2] - z3
                    x_update = clamp(x_pid.update(x_update, sleep=0),max_speed_threshold=50)
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    yaw_update = clamp(yaw_pid.update(yaw_update, sleep=0))
                    drone.send_rc_control(int(x_update*0.8), int(z_update // 2), int(y_update), int(yaw_update//2))
                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x: "+str(round(tvec[i,0,0],2)),(10,40) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"y: "+str(round(tvec[i,0,1],2)),(10,80) ,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    cv2.putText(frame,"z: "+str(round(tvec[i,0,2],2)),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    if abs(tvec[i,0,0]-xi)<eps and abs(tvec[i,0,1]-yi)<eps and abs(tvec[i,0,2]-z3)<eps:
                        f3(drone)
                if s4 and id==4:
                    z4=125
                    rotM=np.zeros(shape=(3,3))          
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    x_update = tvec[0, 0, 0] - xi
                    y_update = -(tvec[0, 0, 1] - yi)
                    z_update = tvec[0, 0, 2] - z4
                    x_update = clamp(x_pid.update(x_update, sleep=0),max_speed_threshold=50)
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    drone.send_rc_control(int(x_update*0.8), int(z_update // 2), int(y_update), int(yaw_update))
                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x:"+str(round(tvec[0,0,0]))+", y:"+str(round(tvec[0,0,1]))+", z:"+str(round(tvec[0,0,2])),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    if abs(tvec[0,0,0]-xi)<eps and abs(tvec[0,0,1]-yi)<eps and abs(tvec[0,0,2]-z4)<eps:
                        f4(drone)
                if s5 and id==5:
                    z5=150
                    rotM=np.zeros(shape=(3,3))          
                    cv2.Rodrigues(rvec[i], rotM, jacobian=0)
                    x_update = tvec[0, 0, 0] - xi
                    y_update = -(tvec[0, 0, 1] - yi)
                    z_update = tvec[0, 0, 2] - z5
                    x_update = clamp(x_pid.update(x_update, sleep=0))
                    y_update = clamp(y_pid.update(y_update, sleep=0))
                    z_update = clamp(z_pid.update(z_update, sleep=0))
                    drone.send_rc_control(int(x_update), int(z_update // 2), int(y_update), 0)
                    frame=cv2.aruco.drawDetectedMarkers(frame,markerCorners,markerIds)
                    frame=cv2.aruco.drawAxis(frame,intrinsic,distortion,rvec[i,:,:],tvec[i,:,:],10)
                    cv2.putText(frame,"x:"+str(round(tvec[0,0,0]))+", y:"+str(round(tvec[0,0,1]))+", z:"+str(round(tvec[0,0,2])),(0,64),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                    if abs(tvec[0,0,0]-xi)<eps and abs(tvec[0,0,1]-yi)<eps and abs(tvec[0,0,2]-z5)<eps:
                        f5(drone)
            
        elif tko:
            print('work')
            drone.send_rc_control(0,-7,0,0)
        else:
            drone.send_rc_control(0,0,0,0)
        cv2.imshow("",frame)
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
