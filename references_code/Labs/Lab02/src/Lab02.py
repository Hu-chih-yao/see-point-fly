import numpy as np
import cv2
#1
image=cv2.imread("p_anya.png")
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image=cv2.GaussianBlur(image,(5,5),0)
#kernel=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
gx=cv2.filter2D(image,-1,kernel)
#kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
kernel=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
gy=cv2.filter2D(image,-1,kernel)
gx=np.array(gx,dtype=np.int32)
gy=np.array(gy,dtype=np.int32)
g=np.sqrt(gx*gx+gy*gy)
g=np.array(g,dtype=np.uint8)
cv2.imshow("p1",g)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("p_anya_output.jpg",g)
#2-a
image=cv2.imread("histogram.jpg")
height=image.shape[0]
width=image.shape[1]
n=height*width
h=np.zeros((256,3))
t=np.zeros((256,3))
for r in range(height):
    for c in range(width):
        for rgb in range (3):
            h[image[r][c][rgb]][rgb]+=1
for p in range(256):
    for rgb in range (3):
        tmp=0
        for j in range(p):
            tmp+=h[j][rgb]
        tmp/=n
        t[p][rgb]=255*tmp
for r in range(height):
    for c in range(width):
        for rgb in range (3):
            image[r][c][rgb]=t[image[r][c][rgb]][rgb]
cv2.imshow("2-a",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("2-a.jpg",image)
#2-b
image=cv2.imread("histogram.jpg")
height=image.shape[0]
width=image.shape[1]
n=height*width
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
h=np.zeros(256)
t=np.zeros(256)
for r in range(height):
    for c in range(width):
        h[hsv[r][c][2]]+=1
for p in range(256):
    tmp=0
    for j in range(p):
        tmp+=h[j]
    tmp/=n
    t[p]=255*tmp
for r in range(height):
    for c in range(width):
        hsv[r][c][2]=t[hsv[r][c][2]]
hsv=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imshow("2-b",hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("2-b.jpg",hsv)
#3
image=cv2.imread("otsu.jpg")
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

height=image.shape[0]
width=image.shape[1]
n=height*width
h=np.zeros((256))
for r in range(height):
    for c in range(width):
        h[image[r][c]]+=1
max_var_b=0
max_threshold=-1
for threshold in range(256):
    lower,higher=np.split(h,[threshold])
    n_b=0
    u_b=0
    for p in range(threshold):
        n_b+=h[p]
        u_b+=h[p]*p
    if n_b!=0:
        u_b/=n_b
    n_o=0
    u_o=0
    for p in range(threshold,256):
        n_o+=h[p]
        u_o+=h[p]*p
    if n_o!=0:
        u_o/=n_o
    var_b=n_b*n_o*(u_b-u_o)**2
    if var_b>max_var_b:
        max_var_b=var_b
        max_threshold=threshold
print(max_var_b)
print(max_threshold)
for r in range(height):
    for c in range(width):
        if image[r][c]<max_threshold:
            image[r][c]=0
        else:
            image[r][c]=255
cv2.imshow("3",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("3.1.jpg",image)
'''
ret,th = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("3",th)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("3.jpg",th)
'''