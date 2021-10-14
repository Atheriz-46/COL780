import os
import cv2 as cv
import re
import numpy as np

def remove_noise(img, filter_size=3,threshold=80):
    kernel = np.ones((filter_size,filter_size),np.uint8)
    _,img = cv.threshold(img, 126, 255, cv.THRESH_BINARY)
    
    img=cv.GaussianBlur(img, (21, 21), 0)
    _,img = cv.threshold(img, 126, 255, cv.THRESH_BINARY)
    
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11)),iterations=2)
    

    img = cv.erode(img, kernel,iterations=1)
    return img
    
def model(args):
    
    backSub = cv.createBackgroundSubtractorKNN(history = 600,dist2Threshold=160, detectShadows = False)#0.3007

    for file in sorted(os.listdir(args.inp_path)):
        if file[-4:] in ['.png','.jpg']:
            frame = cv.imread(os.path.join(args.inp_path,file))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if frame is None:
                break
            
            
            fgMask = None
            
            with open(args.eval_frames, 'r+') as f:
                start,end = map(int,f.read().split())
            frame_no = int(re.search(r'[0-9]+',file)[0])
            
            fgMask = backSub.apply(frame)
            
            if frame_no>=start and frame_no<=end:
                

                cv.imwrite(os.path.join(args.out_path,'gt' + file[2:-3]+'png'), remove_noise(fgMask,3))

            
            
            