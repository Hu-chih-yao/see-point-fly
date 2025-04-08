import cv2
import numpy as np

fs=cv2.FileStorage("cali.xml",cv2.FILE_STORAGE_READ)
intrinsic=fs.getNode("intrinsic").mat()
distortion=fs.getNode('distortion').mat()
fs.release()

len_face=16
len_body_h=180
len_body_w=90

def body(frame):
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame)
    for x,y,w,h in rects:
        ul=(int(x),int(y));     ur=(int(x+w),int(y))
        ll=(int(x),int(y+h));   lr=(int(x+w),int(y+h))
        subtitle=(int(x+2),int(y+h-15))
        frame=cv2.rectangle(frame,ul,lr,(0,255,255),2)
        objp=np.float32([(0,0,0),(len_body_w,0,0),(len_body_w,len_body_h,0),(0,len_body_h,0)])
        imgPoints=np.float32([ul,ur,lr,ll])
        retval,rvec,tvec=cv2.solvePnP(objp, imgPoints, intrinsic, distortion)
        cv2.putText(frame,str(round(float(tvec[2]/100),2))+" m",subtitle,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)

def face(frame):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rects = face_cascade.detectMultiScale(frame,scaleFactor=1.03,minNeighbors=18)
    for x,y,w,h in rects:
        ul=(int(x),int(y));     ur=(int(x+w),int(y))
        ll=(int(x),int(y+h));   lr=(int(x+w),int(y+h))
        subtitle=(int(x+2),int(y+h-15))
        frame=cv2.rectangle(frame,ul,lr,(0,255,0),2)
        objp=np.float32([(0,0,0),(len_face,0,0),(len_face,len_face,0),(0,len_face,0)])
        imgPoints=np.float32([ul,ur,lr,ll])
        retval,rvec,tvec=cv2.solvePnP(objp, imgPoints, intrinsic, distortion)
        cv2.putText(frame,str(round(float(tvec[2]),2))+" cm",subtitle,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

cap=cv2.VideoCapture(0)
while(True):
    ret, frame=cap.read()
    body(frame)
    face(frame)
    cv2.imshow("frame",frame)
    cv2.waitKey(33)