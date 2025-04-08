import numpy as np
import cv2
#GrabCut Algorithm
image=cv2.imread("test.jpg")

rect=cv2.selectROI("roi",image,False,False)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)

mask_new,bgdModel,fgdModel=cv2.grabCut(image,None,rect,bgdModel,fgdModel,15,cv2.GC_INIT_WITH_RECT)
mask_new=np.where((mask_new==0)|(mask_new==2),0,1).astype("uint8")
image=image*mask_new[:,:,np.newaxis]

cv2.imshow("test.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("test_output.jpg",image)
# Background subtraction / Treshold / Connected component
cap=cv2.VideoCapture("train.mp4")
backSub=cv2.createBackgroundSubtractorMOG2()
shadowval=backSub.getShadowValue()
if not cap.isOpened():
    print("not open")
while(True):
    ret,frame=cap.read()
    if ret == False:
        break
    fgmask=backSub.apply(frame)
    ret,nmask=cv2.threshold(fgmask,shadowval,255,cv2.THRESH_BINARY)
    #Connected Component
    T=200
    '''
    #call function
    num_labels,labels,stats,centroids=cv2.connectedComponentsWithStats(nmask,connectivity=4,ltype=None)
    for i in range(num_labels):
        if stats[i][4]>T:
            xu=int(stats[i][1])
            yl=int(stats[i][0])
            xd=int(stats[i][1]+stats[i][3])
            yr=int(stats[i][0]+stats[i][2])
            cv2.rectangle(frame,(yl,xu),(yr,xd),(0,255,0),2)
    cv2.imshow("frame",frame)
    cv2.waitKey(33)
    '''
    #self implement
    height=frame.shape[0]
    width=frame.shape[1]
    n=height*width
    cc=np.zeros((height,width))
    m=np.zeros((n,6))
    cnt=1
    for r in range(height):
        for c in range(width):
            if nmask[r][c]==255:
                if r>0 and nmask[r-1][c]==255:
                    g=int(cc[r-1][c])
                    cc[r][c]=g
                    m[g][0]+=1
                    if r>m[g][4]:
                        m[g][4]=r
                    if c>0 and nmask[r][c-1]==255:
                        g2=int(cc[r][c-1])
                        if g!=g2:
                            if m[g][1]!=0:
                                gm1=m[g][1]
                            else:
                                gm1=g
                            if m[g2][1]!=0:
                                gm2=m[g2][1]
                            else:
                                gm2=g2
                            if gm1>gm2:
                                m[g][1]=gm2
                            else:
                                m[g2][1]=gm1
                elif c>0 and nmask[r][c-1]==255:
                    g=int(cc[r][c-1])
                    cc[r][c]=g
                    m[g][0]+=1
                    if c>m[g][5]:
                        m[g][5]=c
                else:
                    cc[r][c]=cnt
                    m[cnt][0]+=1
                    m[cnt][2]=r
                    m[cnt][3]=c
                    m[cnt][4]=r
                    m[cnt][5]=c
                    cnt+=1
    for i in range (cnt-1,0,-1):
        mp=int(m[i][1])
        if mp==0:
            continue
        m[mp][0]+=m[i][0]
        if(m[i][2]<m[mp][2]):
            m[mp][2]=m[i][2]
        if(m[i][3]<m[mp][3]):
            m[mp][3]=m[i][3]
        if(m[i][4]>m[mp][4]):
            m[mp][4]=m[i][4]
        if(m[i][5]>m[mp][5]):
            m[mp][5]=m[i][5]
    for i in range (cnt):
        if m[i][1]!=0:
            continue
        if m[i][0]>T:
            xu=int(m[i][2])
            yl=int(m[i][3])
            xd=int(m[i][4])
            yr=int(m[i][5])
            cv2.rectangle(frame,(yl,xu),(yr,xd),(0,255,0),2)
    cv2.imshow("frame",frame)
    cv2.waitKey(33)