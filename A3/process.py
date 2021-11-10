import json

with open('negative.json','r+') as f:
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

    }

for name,id_ in file_to_id.items():
    {"file_name": "PennFudanPed/PNGImages/"+name+".png", "height": 536, "width": 559, "id": id_}