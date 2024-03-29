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
import torch
import torchvision

parser = argparse.ArgumentParser(description='This script should use a pretrained HoG detector to make predictions on the provided test set, and store the detections in COCO format in the output file.')
parser.add_argument('--root', type=str, help='path to dataset root directory')
parser.add_argument('--test', type=str, help='path to test json')
parser.add_argument('--out', type=str, help='path to output json')
args = parser.parse_args()


def HOG_Predefined(inp_path = os.path.join('A3','data','PNGImages'),padding=(8, 8),winStride=(4, 4),scale=1.05,probs=None, overlapThresh=0.65,wid=400):
    
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    # inp_path = 'A3\data\PNGImages'
    # files = sorted(os.listdir(inp_path))
    with open('PennFudanPed_val.json','r+') as f:
        data = json.load(f)
    
    # print(files)
    result = []
    for file in data['images']:
        imagePath = file['file_name'].split('/')[-1]
        file_id = file['id']
        if imagePath[-4:]=='.png':

            image = cv.imread(os.path.join(*file['file_name'].split('/')))
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
            if cv.waitKey(10)==27:
                break
    with open('output_1.json','w+') as f:
        json.dump(result,f,indent =2)
    # return result

#HOG_Predefined()


def HOG_train(positive_dset=None,negative_dset=None,inp_path = os.path.join('A3','data','PNGImages')):
    
    # files = sorted(os.listdir(pos_inp_path))
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
# PASCAL_1_to_coco('A3\data\Annotation\FudanPed00016.txt')




def FasterRCNN(inp_path = args.root,test_path = args.test,outputpath=args.out):
    
    class PFDataset(torch.utils.data.Dataset):
        def __init__(self,root,test_data):
            self.root=root
            # self.ids = [x['id'] for x in test_data]
            self.imgs = [x['file_name'] for x in test_data]
        
        def __getitem__(self,idx):

        #    return torch.tensor(np.array(Image.open(os.path.join(self.root,self.imgs[idx])).convert('RGB')).astype('float32').transpose(2,0,1)/255)
        #    uncomment if above fails
            # print(self.imgs[idx])
            # lambda x: x['filename']
            # print(self.imgs[idx])
            image_paths = list(map(lambda x:os.path.join(args.root,*x.split('/')),self.imgs[idx]))
            # print(image)
            images = list(map(lambda image : np.array(Image.open(image).convert('RGB')).astype('float32').transpose(2,0,1)/255, image_paths))
            return torch.tensor(images)
        def __len__(self):
            return len(self.imgs)

    with open(args.test,'r+') as f:
        data = json.load(f)
    dataset = PFDataset(inp_path,data['images'])
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    batch_size = 1
    print('started')
    boxes = []
    for i in range(0,len(dataset),batch_size):
        boxes += model(dataset[i:i+batch_size])
    print(1)
    data = []
    for idx,sample in enumerate(boxes):
        file_id = dataset['images'][idx]['id']
        for i,label in enumerate(sample['labels']):
            if label==1:
                data.append({"image_id": file_id,  
                            "category_id": 1,  
                            "bbox" : sample['boxes'][i].tolist(), 
                                "score" : float( sample['scores'][i])})
    with open(outputpath,'w+') as f:
        json.dump(data,f,indent =2)

FasterRCNN()