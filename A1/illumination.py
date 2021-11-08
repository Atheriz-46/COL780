import os
import cv2 as cv
import re
import numpy as np
from matplotlib import pyplot as plt


def remove_noise(img, filter_size=3,threshold=80):
    img= cv.medianBlur(img,7)
    _,img = cv.threshold(img, 126, 255, cv.THRESH_BINARY)
    d_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
    img = cv.dilate(img, d_kernel,iterations=1)
    e_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    img = cv.erode(img, e_kernel,iterations=1)
    _,img = cv.threshold(img, 126, 255, cv.THRESH_BINARY)
    img= cv.medianBlur(img,9)
    _,img = cv.threshold(img, 126, 255, cv.THRESH_BINARY)  
    kernel = np.ones((filter_size,filter_size),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel,iterations=4)
    img = cv.erode(img, kernel,iterations=1)
    return img

def remove_illumination(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    closed = cv.morphologyEx(img,cv.MORPH_CLOSE,cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)))
    ratio = np.float32(img)/(closed)
    normed = np.uint8(cv.normalize(ratio,ratio,0,255,cv.NORM_MINMAX))
    return normed

def model(args):
    backSub = cv.createBackgroundSubtractorMOG2(history=75,detectShadows = True)
    with open(args.eval_frames, 'r+') as f:
        start,end = map(int,f.read().split())
    
    for file in sorted(os.listdir(args.inp_path)):
        if file[-4:] in ['.png','.jpg']:
            frame_no = int(re.search(r'[0-9]+',file)[0])
            frame = cv.imread(os.path.join(args.inp_path,file))
            frame = remove_illumination(frame)            
            fgMask = backSub.apply(frame)
            fgMask = remove_noise(fgMask,3)
            if frame_no>=start and frame_no<=end:
                cv.imwrite(os.path.join(args.out_path,'gt' + file[2:-3]+'png'),fgMask)

