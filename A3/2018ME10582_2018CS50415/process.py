import json
import os
import cv2 as cv
with open('negatives.json','r+') as f:
    neg = json.load(f)
annotations = []
file_to_id = {}
id_=0
# img_id = 0
for img_id,sample in enumerate(neg):
    if sample["image_id"] not in file_to_id:
        file_to_id[sample["image_id"]] = id_
        id_+=1
    annotations.append({
        "bbox": sample['bbox'], 
        "category_id": 1, 
        "image_id": file_to_id[sample['image_id']], 
        "id": img_id
        })

print(file_to_id)

images=[]
images_d = {}
for name,id_ in file_to_id.items():
    img = cv.imread(os.path.join("PennFudanPed","PNGImages",name.strip()+".png"))
    h,w,_ = img.shape
    images+=[{"file_name": "PennFudanPed/PNGImages/"+name+".png", "height": h, "width": w, "id": id_}]
    images_d[id_]={"h": h, "w": w}

annotation_f = []
for sample in annotations:
    x,y,w,h = sample['bbox']
    w = min(w,images_d[sample['image_id']]['w']-x)
    h = min(h,images_d[sample['image_id']]['h']-y)
    if w*h>10:
        annotation_f.append({
            "bbox": [x,y,w,h], 
            "category_id": sample['category_id'], 
            "image_id": sample['image_id'], 
            "id": sample['id']
            })



with open('negative_fixed.json','w+') as f:
    json.dump({'images':images,'annotations':annotation_f},f,indent =2)