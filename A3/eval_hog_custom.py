'''
define params
'''
k =8
wid=400 
winStride=(4, 4)
scale=1.05
winSize = (16,16)

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
        positive = json.load(f)
    file_name = {x['id'] : x['file_name'] for x in positive['images']}
    pos_data={}
    for sample in data['annotations']:
        bbox = sample['bbox']
        image_path = file_name[sample['image_id']]
        with cv.imread(os.path.join(args.root,*image_path.split('/'))) as image:
            image = imutils.resize(image, width=int(scale*min(wid, image.shape[1])))
            
            x,y,w,h = list(map(int,bbox))
            
            image = image[y:y+h,x:x+w]
            arr = []
            for i in range(0,h-winSize[0],winStride[0]):
                for j in range(0,w-winSize[1],winStride[1]):
                    img = image[i:i+winSize[0],j:j+winSize[1]]
                    arr+= [skimage.feature.hog(image,orientations = 9,block_norm='L2-Hys',feature_vector=True,transform_sqrt=True,channel_axis = 2)]
            pos[sample['id']] = arr    
    return pos  

def get_test_datapoints():
    pass


def HOG_train():
    
    # files = sorted(os.listdir(pos_inp_path))
    pos_sample = get_datapoints(args.train, wid=wid, winStride=winStride, scale=scale, winSize=winSize)
    neg_sample = get_datapoints(args.negative, wid=wid, winStride=winStride, scale=scale, winSize=winSize)
    
    #convert pos_sample,neg_sample to nparray
    X = None

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





  




def HOG_test():
    model = pass 

    val_sample = get_test_datapoints(args.test, wid=wid, winStride=winStride, scale=scale, winSize=winSize)
    for id_, sample in val_sample.items():
        x += [np.bincount(model['kmeans'].predit(sample),minlength=k)]
    x = np.array(x)
    y = model['svm'].predict(x)








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