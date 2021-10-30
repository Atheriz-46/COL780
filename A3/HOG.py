import cv2 as cv
import numpy as np
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
# import skimage
# import sklearn
import re
def HOG_Predefined(inp_path = os.path.join('A3','data','PNGImages'),padding=(8, 8),winStride=(4, 4),scale=1.05,probs=None, overlapThresh=0.65,wid=400):
    
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    # inp_path = 'A3\data\PNGImages'
    files = sorted(os.listdir(inp_path))
    # print(files)
    result = []
    for file_id,imagePath in enumerate(files):
        if imagePath[-4:]=='.png':

            image = cv.imread(os.path.join(inp_path,imagePath))
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
            boxes = [ (rect[i],weights[i]) for i,x in enumerate(rects_n) if x in pick]
            for box in boxes:
                result.append({"image_id": file_id,  
                                "category_id": 1,  
                                "bbox" : box[0], 
                                "score" : box[1]})


            if cv.waitKey(200)==27:
                break

    return result



def HOG_train(inp_path = os.path.join('A3','data','PNGImages'),positive_dset=None):
    
    # files = sorted(os.listdir(pos_inp_path))
    pos_images=[]
    for sample in positive_dset:
        image = cv.imread(os.path.join(pos_inp_path,sample["image_id"]+'.png'))
        x,y,w,h = sample['bbox']
        pos_images.append(image[x:x+h,y:y+h])
    for imagePath in files:
        if imagePath[-4:]=='.png':
            pos_images.append(cv.imread(os.path.join(pos_inp_path,imagePath)))
    files = sorted(os.listdir(neg_inp_path))
    pos_data = preprocess(pos_images)
    neg_images=[]
    for imagePath in files:
        if imagePath[-4:]=='.png':
            neg_images.append(cv.imread(os.path.join(neg_inp_path,imagePath)))
    neg_data = preprocess(neg_images)
    ##Train SVM
    ##Hard Negative Mining
    ##Testing code and generating output

# HOG_Predefined()


def preprocess(images,size = (400,600)):
    # skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None, *, channel_axis=None)
    arr = []
    for image in images:
        image = cv.resize(image,size,cv.INTER_AREA)
        arr+= [skimage.feature.hog(image,orientations = 9,block_norm='L2-Hys',feature_vector=True,transform_sqrt=True,channel_axis = 2)]
    return np.array(arr)


def PASCAL_1_to_coco(path):
    
    with open(path,'r+') as file:
        lines = file.read()
        Image_filename = re.findall('\/([^\/_]+)\.png',lines)[0]
        print(Image_filename)
        # Image_filename = lines[1].split(':')[1].strip().split('/')[-1][:-5]
        data = []
        boxes = re.findall('\(([0-9]+), ([0-9]+)\) - \(([0-9]+), ([0-9]+)\)',lines)
        # print(boxes)
        boxes = list(map(lambda x: [int(x[0]),int(x[1]),int(x[2])-int(x[0]),int(x[3])-int(x[1])],boxes))

        for box in boxes:
            data.append({"image_id": Image_filename,  
                            "category_id" : 1,  
                            "bbox" : box})
    return data 
# PASCAL_1_to_coco('A3\data\Annotation\FudanPed00016.txt')