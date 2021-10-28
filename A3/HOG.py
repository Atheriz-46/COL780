import cv2 as cv
import numpy as np
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
def HOG_Predefined(inp_path = 'A3\data\PNGImages',padding=(8, 8),winStride=(4, 4),scale=1.05,probs=None, overlapThresh=0.65,wid=400)
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    # inp_path = 'A3\data\PNGImages'
    files = sorted(os.listdir(inp_path))
    # print(files)
    for imagePath in files:
        if imagePath[-4:]=='.png':
            image = cv.imread(os.path.join(inp_path,imagePath))
            # cv.imshow("Orig", image)

            image = imutils.resize(image, width=min(wid, image.shape[1]))
            # orig = image.copy()
            try:
                (rects, weights) = hog.detectMultiScale(image, winStride=winStride, padding=padding, scale=scale)
            except:
                print(imagePath)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=probs, overlapThresh=overlapThresh)
            for (xA, yA, xB, yB) in pick:
                cv.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv.imshow("Dectector", image)

            if cv.waitKey(30)=='q':
                break
