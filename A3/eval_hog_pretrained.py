import cv2 as cv
import numpy as np
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
# import skimage
# import sklearn
import re
# import torchvision,torch
from PIL import Image
import json

import argparse

parser = argparse.ArgumentParser(description='This script should use a pretrained HoG detector to make predictions on the provided test set, and store the detections in COCO format in the output file.')
parser.add_argument('--root', type=str, help='path to dataset root directory')
parser.add_argument('--test', type=str, help='path to test json')
parser.add_argument('--out', type=str, help='path to output json')
args = parser.parse_args()

def HOG_Predefined(padding=(8, 8),winStride=(4, 4),scale=1.05,probs=None, overlapThresh=0.65,wid=400):
    
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    # inp_path = 'A3\data\PNGImages'
    # files = sorted(os.listdir(inp_path))
    # with open('PennFudanPed_val.json','r+') as f:
    with open(args.test,'r+') as f:
        data = json.load(f)
    
    # print(files)
    result = []
    for file in data['images']:
        imagePath = file['file_name'].split('/')[-1]
        file_id = file['id']
        if imagePath[-4:]=='.png':

            image = cv.imread(os.path.join(args.root,*file['file_name'].split('/')))
            # cv.imshow("Orig", image)

            image = imutils.resize(image, width=min(wid, image.shape[1]))
            # orig = image.copy()
            (rects, weights) = hog.detectMultiScale(image, winStride=winStride, padding=padding, scale=scale)
            # print(weights)
            rects_n = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects_n, probs=probs, overlapThresh=overlapThresh)
            for (xA, yA, xB, yB) in pick:
                cv.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv.imshow("Detector", image)
            boxes = [ (rects[i].tolist(),weights[i].tolist()) for i,x in enumerate(rects_n) if x in pick]
            for box in boxes:
                result.append({"image_id": file_id,  
                                "category_id": 1,  
                                "bbox" : box[0], 
                                "score" : box[1][0]})

            # print(list(map(type,result[-1].values())))
            # print(result[-1].values())
            # exit(0)
            if cv.waitKey(1)==27:
                break
    # with open('output_1.json','w+') as f:
    with open(args.out,'w+') as f:
        json.dump(result,f,indent =2)
    # return result


params = {}

HOG_Predefined(**params)

