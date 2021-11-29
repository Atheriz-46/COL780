from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import matplotlib.pyplot as plt
import json
import numpy as np

parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--gt', type=str, help='path to ground truth annotations')
parser.add_argument('--pred', type=str, help='path to predicted detections')
args = parser.parse_args()

temp = 'temp.json'
with open(args.pred,'r') as f:
    d_list = json.load(f)

with open('PennFudanPed_val.json','r') as f:
    od_list = json.load(f)

thres_l = np.linspace(0.0,1,100)
def eval(temp_file):
    anno = COCO(args.gt)
    pred = anno.loadRes(temp_file) 
    is_coco = True
    test_indices = range(len(anno.imgs))

    eval = COCOeval(anno, pred, 'bbox')
    if is_coco:
        eval.params.imgIds = test_indices

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    AP = eval.stats[0]
    AR_at_1 = eval.stats[6]
    AR_at_10 = eval.stats[7]
    return AP,AR_at_1,AR_at_10


mr_list = []
fppi_list = []
for thres in thres_l:
    if thres==1:
        break
    n_d = []
    for d in d_list:
        if d['score']>=thres:
            n_d.append(d)
    with open(temp,'w') as f:
        json.dump(n_d,f,indent=2)
    a,b,c = eval(temp)
    mr = 1-c
    fppi = len(n_d)*(1-a)/len(od_list['images'])
    # fppi = (1-a)
    if fppi<1:
        mr_list.append(mr)
        fppi_list.append(fppi)

plt.figure()
plt.plot(fppi_list,mr_list,color='blue')
plt.xlabel("FPPI")
plt.ylabel("Miss Rate")
plt.title('FPPI-Miss rate Detection curve')
plt.savefig(args.pred[:-4]+'png')
    


    
    