'''
define params
'''
k =8
wid=400 
winStride=(4, 4)
scale=1.05
winSize = (64,16)

##################################################################################

import cv2 as cv
import numpy as np
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import skimage
import sklearn
import re
# import torchvision,torch
from PIL import Image
import json


import argparse

parser = argparse.ArgumentParser(description='This script should use a pretrained HoG detector to make predictions on the provided test set, and store the detections in COCO format in the output file.')
parser.add_argument('--root', type=str, help='path to dataset root directory')
parser.add_argument('--train', type=str, help='path to train json')
parser.add_argument('--negative',type=str, help='path to negative sample json')
parser.add_argument('--test', type=str, help='path to test json')
parser.add_argument('--out', type=str, help='path to output json')
parser.add_argument('--model', type=str, help='path to trained SVM model')
args = parser.parse_args()



def get_datapoints(data_path,**kwargs):
    with open(data_path,'r+') as f:
        data = json.load(f)
    file_name = {x['id'] : x['file_name'] for x in data['images']}
    pos_data={}
    for sample in data['annotations']:
        bbox = sample['bbox']
        image_path = file_name[sample['image_id']]
        with cv.imread(os.path.join(args.root,*image_path.split('/'))) as image:
            # image = imutils.resize(image, width=int(scale*min(wid, image.shape[1])))
            
            x,y,w,h = list(map(int,bbox))
            
            image = image[y:y+h,x:x+w]
            arr = []
            for i in range(0,h-winSize[0],winStride[0]):
                for j in range(0,w-winSize[1],winStride[1]):
                    img = image[i:i+winSize[0],j:j+winSize[1]]
                    arr+= [skimage.feature.hog(img,orientations = 9,block_norm='L2-Hys',feature_vector=True,transform_sqrt=True,channel_axis = 2)]
            pos_data[sample['id']] = arr    
    return pos_data  

def get_test_datapoints(data_path,kmeans,**kwargs):
    with open(data_path,'r+') as f:
        data = json.load(f)
    # file_name = {x['id'] : x['file_name'] for x in positive['images']}
    # pos_data={}
    for sample in data['images']:
        # bbox = sample['bbox']
        id_ = sample['id']
        image_path = sample['file_name']
        with cv.imread(os.path.join(args.root,*image_path.split('/'))) as image:
            # image = imutils.resize(image, width=int(scale*min(wid, image.shape[1])))
            
            # x,y,w,h = list(map(int,bbox))
            w,h = sample['width'],sample['height']
            # image = image[y:y+h,x:x+w]
            # arr = {}
            idy,idx = int((h-winSize[0])//winStride[0]),int((w-winSize[1])//winStride[1])
            arr = np.zeros((idy,idx))
            for i in range(idy):
                for j in range(idx):
                    img = image[i*winStride[0]:i*winStride[0]+winSize[0],j*winStride[1]:j*winStride[1]+winSize[1]]
                    x = skimage.feature.hog(img,orientations = 9,block_norm='L2-Hys',feature_vector=True,transform_sqrt=True,channel_axis = 2)
                    arr[i,j] = kmeans.predict(x)[0]

            # pos[sample['id']] = arr    
        yield arr,id_
    # return pos  
def convert_dict_to_linear(*args):
    res = []
    for d in args:
        for _,v in d:
            res+=v
    return res

def HOG_train():
    
    # files = sorted(os.listdir(pos_inp_path))
    pos_sample = get_datapoints(args.train, wid=wid, winStride=winStride, scale=scale, winSize=winSize)
    neg_sample = get_datapoints(args.negative, wid=wid, winStride=winStride, scale=scale, winSize=winSize)
    
    #convert pos_sample,neg_sample to nparray
    X = convert_dict_to_linear(pos_sample,neg_sample)

    premodel = sklearn.cluster.KMeans(n_clusters=k,init='k-means++', n_init = 10, max_iter=300, tol=0.0001).fit(X)
    del X 
    x = []
    y=[]
    for id_, sample in pos_sample.items():
        x += [np.bincount(premodel.predit(sample),minlength=k)]
        y+=[[1]]
    for id_, sample in neg_sample.items():
        x += [np.bincount(premodel.predit(sample),minlength=k)]
        y+=[[0]]
    # y = np.array([[1]]*len(pos_sample)+ [[0]]*len(neg_sample))
    x,y = np.array(x),np.array(y)
    assert len(x)==len(y)
    np.rand.seed(20)
    p = np.random.permutation(len(x))
    x,y = x[p],y[p]


    svm = sklearn.svm.SVC()
    svm.fit(x,y)

    model={'svm':svm,'kmeans':premodel}
    with open(args.model,'w+') as f:
        pickle.save(model,f)





  



def get_bbox(dp,svm,confidenceTreshold=0.7):
    bbox,scores = [],[]
    for y in range(dp.shape[0]):
        for x in range(dp.shape[1]):
            for h in range(dp.shape[0]-y):
                for w in range(dp.shape[1]-x):
                    fv = np.bincount(dp[y:y+h,x:x+w],minlength=k)
                    if svm.predict(fv)==1:
                        s = svm.decision_function(fv)
                        if s>confidenceTreshold:
                            bbox.append([x*winStride[1],y*winStride[0],w*winStride[1]+winSize[1],h*winStride[0]+winSize[0]])  
                            scores.append([s])

    return bbox, scores



def HOG_test(probs=None, overlapThresh=0.07):

    # Read model from
    with open(args.model, 'r+') as f:
        model = pickle.load(f) 

    # val_sample = 
    # for id_, sample in val_sample.items():
        # x += [np.bincount(model['kmeans'].predit(sample),minlength=k)]
    for image,file_id in get_test_datapoints(args.test,model['kmeans'],wid=wid, winStride=winStride, scale=scale, winSize=winSize):
        bbox,scores = get_bbox(image,model['svm'])
        bbox_n = np.array([[x, y, (x + w), (y + h)] for (x, y, w, h) in bbox])

        pick = non_max_suppression(bbox, probs=probs, overlapThresh=overlapThresh)
        boxes,weight = [],[]
        for i,x in enumerate(bbox_n):
            if x in pick:
                x,y,w,h = bbox[i].tolist()
                if w*h>area:
                    boxes+=[[x+w*s[0],y+h*s[0],w*(1-2*s[0]),h*(1-2*s[0])]]
                else:
                    boxes+=[[x+w*s[1],y+h*s[1],w,h]]
                    # boxes+=[[x+w*s[1],y+h*s[1],w*(1-2*s[1]),h*(1-2*s[1])]]
                # print(boxes)
                weight+=[weights[i]]
                # weight.append(weights[i])
        
        weight = np.array(weight)

        # to activate softmax
        # weight = np.exp(weight)
        weight = (weight+k)/np.sum(weight+k)
        
        for box,w in zip(boxes,weight):
            # print(file_id)
            result.append({"image_id": file_id,  
                            "category_id": 1,  
                            "bbox" : box, 
                            "score" : w[0]
                            })

        # print(list(map(type,result[-1].values())))
        # print(result[-1].values())
        # exit(0)
        # if cv.waitKey(1)==27:
            # break
    # with open('output_1.json','w+') as f
    # print(result)

    with open(args.out,'w+') as f:
        json.dump(result,f,indent =2)
        # for (xA, yA, xB, yB) in pick:
            # cv.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # cv.imshow("Detector", image)








'''



    pos_images=[]
    for sample in positive_dset:
        image = cv.imread(os.path.join(pos_inp_path,sample["image_id"]+'.png'))
        x,y,w,h = sample['bbox']
        pos_images.append(image[y:y+h,x:x+h])
    # for imagePath in files:
    #     if imagePath[-4:]=='.png':
    #         pos_images.append(cv.imread(os.path.join(pos_inp_path,imagePath)))
    # files = sorted(os.listdir(neg_inp_path))
    pos_data = preprocess(pos_images)
    neg_images=[]
    for sample in negative_dset:
        image = cv.imread(os.path.join(pos_inp_path,sample["image_id"]+'.png'))
        # if imagePath[-4:]=='.png':
        x,y,w,h = sample['bbox']
        neg_images.append(image[y:y+h,x:x+h])
        # _images.append()
    # neg_data = preprocess(neg_images)
    ##Get sliding window images of neg_data

    neg_data_extended = generate_subsets(neg_images)
    neg_data_extended = preprocess(neg_data_extended)

    ##Train SVM
    svm=sklearn.svm.SVC(probability=True,random_state=10)
    # svm=sklearn.linear_model.SGDClassifier(probability=True,random_state=10,warm_start=True)
    svm.fit(np.vstack((pos_data,neg_data)),np.vstack((np.ones((len(pos_data),1)),np.zeros((len(neg_data),1)))))
    

    ##Hard Negative Mining
    # for i in range(3):
    #     result=svm.predit(neg_data_extended)
    #     data = neg_data_extended[result==1]
    #     svm.fit(data,np.zeros((len(neg_data),1)))

    ##Testing code and generating output
    with open('model.sav','w+') as f:
        pickle.save(svm,f)

def generate_subsets(images,scale=[2,2],stride=[5,5],kernel=[10,10]):
    
    result = []
    for image in images:
        w=kernel[0]
        while w<len(image[0]):
            h=kernel[1]
            while h<len(image):
                for x in range(0,len(image[0])-w,stride[0]):
                    for y in range(0,len(image)-h,stride[1]):
                        result+=[image[y:y+h,x:x+w]]
                h*=scale[1]
            w*=scale[0]
    return result
    

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
# PASCAL_1_to_coco('A3\data\Annotation\FudanPed00016.txt')'''