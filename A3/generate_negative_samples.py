import cv2 as cv
import os
import re
import json


inp_path = 'A3\data\PNGImages'
files = sorted(os.listdir(inp_path))
outputpath = input('Where do you want to store the results?')
# result = []
image=None
global startingPoint, endPoint,data,Image_filename 
startingPoint, endPoint,data,Image_filename= None,None,[],''
def selectRectangle(event, x, y, flags, param):
    # global startingPoint, endPoint, bbox
    global startingPoint, endPoint,data,Image_filename 

    if event == cv.EVENT_LBUTTONDOWN:
        endPoint = None
        startingPoint = [x,y]
    elif event == cv.EVENT_LBUTTONUP:
        endPoint = [x,y]
        cv.rectangle(image, (startingPoint[0], startingPoint[1]), (endPoint[0], endPoint[1]),  (0,0,255), 2)
        data.append({"image_id": Image_filename,  
        "category_id" : 0,  
            "bbox" : startingPoint+[endPoint[0]-startingPoint[0],endPoint[1]-startingPoint[1]]})
        cv.imshow("Mark", image)
        cv.waitKey(0)

def PASCAL_1_to_coco(path):
    
    with open(path,'r+') as file:
        lines = file.read()
        Image_filename = re.findall('\/([^\/_]+)\.png',lines)[0]
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
for file_id,imagePath in enumerate(files):
    if imagePath[-4:]=='.png':
        # bbox = []
        Image_filename = imagePath[:-4]
        image = cv.imread(os.path.join(inp_path,imagePath), cv.IMREAD_UNCHANGED)
        pos = PASCAL_1_to_coco(os.path.join('A3','data','Annotation',Image_filename+'.txt'))
        for box in pos:
            x,y,w,h = box['bbox']
            cv.rectangle(image, (x, y),(x+w,y+h),  (0,255,0), 2)

        # frame11 = cv.imread(imgs[0], cv2.IMREAD_UNCHANGED)
        cv.namedWindow("Mark")
        cv.setMouseCallback("Mark", selectRectangle)
        cv.imshow("Mark", image)
        if cv.waitKey(0) == 27:
            break
        cv.destroyAllWindows()
        
        # for box in boxes:
        #     result.append({"image_id": file_id,  
        #                     "category_id": 1,  
        #                     "bbox" : box[0], 
        #                     "score" : box[1]})


        # if cv.waitKey(200)==27:
            # break
with open(outputpath,'w+') as f:
    json.dump(data,f,indent =2)
print(data)