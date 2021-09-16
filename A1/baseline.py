import os
import cv2 as cv
import re

def remove_noise(img, filter_size=3,threshold=80):
    kernel = np.ones((filter_size,filter_size),np.uint8)

    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel,iterations=1)
    return img
    
    # img= np.where(img<threshold,0,255)
    
    return img
def model(args):
    
    # backSub = cv.createBackgroundSubtractorMOG2()#0.2398
    backSub = cv.createBackgroundSubtractorKNN()#0.3007

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
            # if frame_no>=start and frame_no<=end:
            #     fgMask = backSub.apply(frame)
            # else:
            #     fgMask = np.zeros(frame.shape, dtype = "uint8")
            
            # cv.imwrite(os.path.join(args.out_path,'gt' + file[2:]), fgMask)
            fgMask = backSub.apply(frame)
            
            if frame_no>=start and frame_no<=end:
                
                cv.imwrite(os.path.join(args.out_path,'gt' + file[2:-3]+'png'), remove_noise(fgMask,3))

            
            
            # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
            #         cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            
            
            # cv.imshow('Frame', frame)
            # keyboard = cv.waitKey(30)
            # if keyboard == 'q' or keyboard == 27:
            #     break