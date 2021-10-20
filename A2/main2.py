import cv2 as cv
import numpy as np
import math
# import glob
import os
import re
# Warp function for affine
def getWfromP(p):
    W = np.array([[1+p[0,0],p[0,2],p[0,4]],
                    [p[0,1], 1+p[0,3],p[0,5]]])
    return W 

# Wrapping the image
def wrappingFunction(I,W,Tpoints):
    n,it = Tpoints.shape
    transformedImagePoints = np.empty([2,n])
    transformedImagePoints = np.matmul(W,Tpoints.T)
    transformedImagePoints = transformedImagePoints.T
    transformedImageIntensities = np.empty([n,1])
    i = transformedImagePoints.astype(int)
    a = transformedImagePoints - i
    # a,b = x-i,y-j
    transformedImageIntensities[:,0] = bilinear_interpolate(transformedImagePoints,I)
    # (1-a[:0])I[i[:,1],i[:,0]]
    # transformedImageIntensities[:,0] = I[transformedImagePoints[:,1].astype(int),transformedImagePoints[:,0].astype(int)]
    return transformedImagePoints,transformedImageIntensities

def bilinear_interpolate(transformedImagePoints,I):
    i = transformedImagePoints.astype(int)
    a = transformedImagePoints - i
    return (1-a[:,0])*((1-a[:,1])*I[i[:,1],i[:,0]]+a[:,1]*I[i[:,1]+1,i[:,0]]) + a[:,0]*((1-a[:,1])*I[i[:,1],i[:,0]+1]+a[:,1]*I[i[:,1]+1,i[:,0]+1])
# Wrapping the gradient image
def wrappingFunctionOfGrad(gradientX, gradientY ,IWpoints):
    n,it = IWpoints.shape
    gradXIntensities = np.empty([n,1])
    gradYIntensities = np.empty([n,1])
    gradXIntensities[:,0] = bilinear_interpolate(IWpoints,gradientX)#[IWpoints[:,1].astype(int),IWpoints[:,0].astype(int)]
    gradYIntensities[:,0] = bilinear_interpolate(IWpoints,gradientY)
    return gradXIntensities, gradYIntensities

# Calculating change in parameters p
def clacChangeInParams(error, IWdx, IWdy, TPoints):
    img1 = IWdx[:,0] * [TPoints[:,0]]
    img2 = IWdx[:,0] * [TPoints[:,1]]
    img3 = IWdy[:,0] * [TPoints[:,0]]
    img4 = IWdy[:,0] * [TPoints[:,1]]
    dIW = np.hstack((img1.T,img3.T,img2.T,img4.T,IWdx,IWdy))
    sumP = np.matmul(dIW.T,error)
    sumHess = np.matmul(dIW.T,dIW)
    sumP = np.matmul(np.linalg.pinv(sumHess), sumP)
    return sumP.T

# LucasKanadeTracker implementation
def lucasKanadeTracker(Tpoints, Tintensity, I, p):# startingPoint, endPoint):
    threshold = 0.07
    changeP = 100
    gradientX = cv.Sobel(I,cv.CV_64F,1,0,ksize=3)
    gradientY = cv.Sobel(I,cv.CV_64F,0,1,ksize=3)
    it = 0
    safeW,safep = getWfromP(p),p 
    while(changeP > threshold):
        print(changeP)
        it += 1
        W = getWfromP(p)
        IWpoints, IWi = wrappingFunction(I,W,Tpoints)
        error = Tintensity - IWi
        IWdx, IWdy = wrappingFunctionOfGrad(gradientX, gradientY ,IWpoints)
        deltaP= clacChangeInParams(error, IWdx, IWdy,Tpoints)
        changeP = np.linalg.norm(deltaP)
        p = p + deltaP
        # p[0,0] += deltaP[0,0]
        # p[0,1] += deltaP[1,0]
        # p[0,2] += deltaP[2,0]
        # p[0,3] += deltaP[3,0]
        # p[0,4] += deltaP[4,0]        
        # p[0,5] += deltaP[5,0]
        # newStart = np.array([[startingPoint[0]],[startingPoint[1]],[1]])
        # newend = np.array([[endPoint[0]],[endPoint[1]],[1]])
        # s = np.matmul(W,newStart)
        # e = np.matmul(W,newend)
        if (it > 10):
            return safeW,safep 
    return W,p

# LucasKanadeTracker implementation
def pyramidLucasKanade(frame1Points, frame1Intensities, frame2, p, startingPoint, endPoint):
    W,p = lucasKanadeTracker(frame1Points, frame1Intensities, frame2, p, startingPoint, endPoint)
    return W,p

# Selecting template from the image
def selectRectangle(event, x, y, flags, param):
    global startingPoint, endPoint
    if event == cv.EVENT_LBUTTONDOWN:
        startingPoint = [x,y]
    elif event == cv.EVENT_LBUTTONUP:
        endPoint = [x,y]
        cv.rectangle(frame11, (startingPoint[0], startingPoint[1]), (endPoint[0], endPoint[1]),  (255,255,255), 2)
        cv.imshow("Mark", frame11)
        cv.waitKey(0)

def split(box,image):
    pts = []
    intensity = []
    for y in range(box[3]+1):
        for x in range(box[2]+1):
            pts.append([box[0]+x,box[1]+y,1])
            intensity.append(image[box[1]+y,box[0]+x])
    return np.array(pts),np.array(intensity)

def lk_tracker(dir,outfile):
    inp_path = os.path.join(dir,'img')
    gt = np.genfromtxt(os.path.join(dir,'groundtruth_rect.txt'), delimiter=',')
    gt = np.int32(gt)
    box = gt[0]
    # box = [227,207,122,99]
    gt = gt[1:]
    print(gt.shape)
    files = sorted(os.listdir(inp_path))
    template = cv.imread(os.path.join(inp_path,files[0]))
    # files = files[1:]
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    Tpoints, Tintensity = split(box,template)
    # print(template.shape)
    # return
    output = []
    sIOU,n = 0.,0.
    # tracker = LK(template,box)
    for file in files:
        if file[-4:] in ['.jpg']:
            frame = cv.imread(os.path.join(inp_path,file))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if frame is None: break             
            frame_no = int(re.search(r'[0-9]+',file)[0])

            p = np.zeros([1,6], dtype = float)
            print('hey')

            w,p = lucasKanadeTracker(Tpoints, Tintensity, frame, p)#, startingPoint, endPoint)
            # box_t = tracker.fit(frame)
            # print(len(box_t))
            print('hey')
            output.append(box_t)

            ''' Score '''

            sIOU += IOU(box_t,gt[frame_no-1])
            n+=1.
            
            # frame = cv.rectangle(frame,(box_t[1],box_t[0]) ,(box_t[1]+box_t[3],box_t[2]+box_t[0]) , (255,0,0), 2)
            frame = cv.rectangle(frame,(box_t[0],box_t[1]) ,(box_t[0]+box_t[2],box_t[3]+box_t[1]) , (255,0,0), 2)
            # frame = cv.rectangle(frame,(gt[frame_no-1,1],gt[frame_no-1,0]) ,(gt[frame_no-1,1]+gt[frame_no-1,3],gt[frame_no-1,2]+gt[frame_no-1,0]) , (0,255,0), 2)
            frame = cv.rectangle(frame,(gt[frame_no-1,0],gt[frame_no-1,1]) ,(gt[frame_no-1,0]+gt[frame_no-1,2],gt[frame_no-1,3]+gt[frame_no-1,1]) , (0,255,0), 2)
            cv.imshow('Image',frame)
            # cv.waitKey(0)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            

    
    np.savetxt(outfile, np.array(output), delimiter=",",fmt='%d')
    # print('mIOU score: {0:.4f}'.format(sIOU/n))
lk_tracker('.\A2\data\BlurCar2','.\A2\data\BlurCar2\outfile')
