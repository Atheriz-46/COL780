'''
define params
'''
k =300
wid=400 
winStride=(16, 16)
scale=1.05
winSize = (32,64)

##################################################################################

import cv2 as cv
import numpy as np
import os
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils
import skimage
import skimage.feature
import sklearn
import sklearn.svm
import sklearn.cluster
import re
import pickle
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
        image = cv.imread(os.path.join(args.root,*image_path.split('/')))
            # image = imutils.resize(image, width=int(scale*min(wid, image.shape[1])))
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)   
        x,y,w,h = list(map(int,bbox))
        
        image = image[y:y+h,x:x+w]
        arr = []
        for i in range(0,h-winSize[0],winStride[0]):
            for j in range(0,w-winSize[1],winStride[1]):
                img = image[i:i+winSize[0],j:j+winSize[1]]
                hog = skimage.feature.hog(img,orientations = 9,block_norm='L2',feature_vector=True,transform_sqrt=True)
                arr+= [hog]
                # print(hog)
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
        image = cv.imread(os.path.join(args.root,*image_path.split('/')))
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
                x = skimage.feature.hog(img,orientations = 9,block_norm='L2',feature_vector=True,transform_sqrt=True)
                arr[i,j] = kmeans.predict(x.reshape(1,-1))[0]

            # pos[sample['id']] = arr 
        # print(arr)   

        yield arr,id_
    # return pos  
def convert_dict_to_linear(d1,batch):
    i = 0
    res =[]
    for _,v in d1.items():
        res+=v
        i+=1
        if i%batch==0:
            yield res
            res = []
    # return res
    yield res

def HOG_train():
    
    # files = sorted(os.listdir(pos_inp_path))
    pos_sample = get_datapoints(args.train, wid=wid, winStride=winStride, scale=scale, winSize=winSize)
    
    #convert pos_sample,neg_sample to nparray
    # X = np.array(convert_dict_to_linear(pos_sample,neg_sample))
    # print(pos_sample,neg_sample)
    # premodel = sklearn.cluster.KMeans(n_clusters=k,init='k-means++', n_init = 10, max_iter=300, tol=0.0001).fit(X)
    premodel = sklearn.cluster.MiniBatchKMeans(n_clusters=k,init='k-means++', n_init = 10, max_iter=300, tol=0.0001)
    for X in convert_dict_to_linear(pos_sample,32):
        
        premodel = premodel.partial_fit(np.array(X))
    neg_sample = get_datapoints(args.negative, wid=wid, winStride=winStride, scale=scale, winSize=winSize)
    
    for X in convert_dict_to_linear(neg_sample,32):
        premodel = premodel.partial_fit(np.array(X))
    del X 
    x = []
    y=[]
    for id_, sample in pos_sample.items():
        # x += [np.bincount(premodel.predict(np.array(sample)),minlength=k)]
        # print(sample)
        if sample:
            temp=premodel.predict(np.array(sample))
            # print(temp)
            x += [np.bincount(temp,minlength=k)]
            y+=[[1]]
    print('Positive done')
    for id_, sample in neg_sample.items():
        if sample:
            temp=premodel.predict(np.array(sample))
            # print(temp)
            x += [np.bincount(temp,minlength=k)]
            y+=[[0]]
    print('Negative done')
    
    # y = np.array([[1]]*len(pos_sample)+ [[0]]*len(neg_sample))
    x,y = np.array(x),np.array(y)
    assert len(x)==len(y)
    # np.rand.seed(20)
    p = np.random.permutation(len(x))
    x,y = x[p],y[p].ravel()


    svm = sklearn.svm.SVC(probability=True)
    svm.fit(x,y)

    model={'svm':svm,'kmeans':premodel}
    with open(args.model,'wb') as f:
        pickle.dump(model,f)






  



def get_bbox(dp,svm,confidenceTreshold=0.7,minarea = 4):
    bbox,scores = [],[]
    for y in range(dp.shape[0]):
        for x in range(dp.shape[1]):
            for h in range(dp.shape[0]-y):
                for w in range(dp.shape[1]-x):
                    temp = dp[y:y+h+1,x:x+w+1].ravel().astype('int64')
                    fv = np.bincount(temp,minlength=k).reshape(1,-1)
                    if svm.predict(fv)==1:
                        s = svm.predict_proba(fv)[0][1]
                        # print(s)
                        # print(s)
                        if s>confidenceTreshold:# and h*w > minarea:
                        # if s>confidenceTreshold:
                            bbox.append([x*winStride[1],y*winStride[0],w*winStride[1]+winSize[1],h*winStride[0]+winSize[0]])  
                            scores.append([s])
                        # print(x,y,w,h)

    return bbox, scores



def HOG_test(probs=None, overlapThresh=0.001,area=0,s=[0.1,0],threshold=0.3):
    result = []
    # Read model from
    with open(args.model, 'rb') as f:
        model = pickle.load(f) 

    # val_sample = 
    # for id_, sample in val_sample.items():
        # x += [np.bincount(model['kmeans'].predit(sample),minlength=k)]
    for image,file_id in get_test_datapoints(args.test,model['kmeans'],wid=wid, winStride=winStride, scale=scale, winSize=winSize):
        bbox,weights = get_bbox(image,model['svm'])
        bbox_n = np.array([[x, y, (x + w), (y + h)] for (x, y, w, h) in bbox])
        # print(bbox[0])
        pick = non_max_suppression(np.array(bbox_n), probs=probs, overlapThresh=overlapThresh)
        boxes,weight = [],[]
        for i,x in enumerate(bbox_n):
            if x in pick:
                x,y,w,h = bbox[i]
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
        # weight = (weight+k)/np.sum(weight+k)
        
        for box,w in zip(boxes,weight.tolist()):
            # print(file_id)
            # if w[0]>
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




if args.train:
    HOG_train()
else: 
    HOG_test()
