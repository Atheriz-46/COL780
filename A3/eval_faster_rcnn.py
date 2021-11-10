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
import torch
import torchvision

parser = argparse.ArgumentParser(description='This script should use a pretrained HoG detector to make predictions on the provided test set, and store the detections in COCO format in the output file.')
parser.add_argument('--root', type=str, help='path to dataset root directory')
parser.add_argument('--test', type=str, help='path to test json')
parser.add_argument('--out', type=str, help='path to output json')
args = parser.parse_args()



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
    res = []
    
    for idx in range(0,len(dataset),batch_size):
        # print(idx)
        sample = model(dataset[idx:idx+batch_size])[0]
        file_id = data['images'][idx]['id']
        for i,label in enumerate(sample['labels']):
            if label==1:
                res.append({"image_id": file_id,  
                            "category_id": 1,  
                            "bbox" : sample['boxes'][i].tolist(), 
                            "score" : float( sample['scores'][i])})
    print(1)
    # for idx,sample in enumerate(boxes):
        
    with open(outputpath,'w+') as f:
        json.dump(res,f,indent =2)

FasterRCNN()