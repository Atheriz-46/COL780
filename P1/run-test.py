# Evaluation 
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer

from __future__ import print_function

import os
import faiss
import numpy as np

from PIL import Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# from model import ReidModel, DummyModel
from utils import get_id
from metrics import rank1, rank5, calc_ap

from TransReid.model.make_model import make_model
 
# ### Set feature volume sizes (height, width, depth) 
# TODO: update with your model's feature length

batch_size = 1
H, W, D = 7, 7, 2048 # for dummymodel we have feature volume 7x7x2048

# ### Load Model
from TransReid.config import cfg
import argparse
# TODO: Uncomment the following lines to load the Implemented and trained Model

parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()
# save_path = "<model weight path>"
model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
model.load_state_dict(torch.load(save_path), strict=False)
model.eval()

# TODO: Comment out the dummy model
# model = DummyModel(batch_size, H, W, D)

# ### Data Loader for query and gallery

# TODO: For demo, we have resized to 224x224 during data augmentation
# You are free to use augmentations of your own choice
# transform_query_list = [
#         transforms.Resize((224,224), interpolation=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]
# transform_gallery_list = [
#         transforms.Resize(size=(224,224), interpolation=3), #Image.BICUBIC
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]

# data_transforms = {
#         'query': transforms.Compose( transform_query_list ),
#         'gallery': transforms.Compose(transform_gallery_list),
#     }
from TransReid.dataset.make_dataloader import make_dataloader
train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
# image_datasets = {}
# image_datasets['query']=val_loader[:num_query]
# image_datasets['gallery']=val_loader[num_query:]
# data_dir = "data/val"

# image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
#                                           data_transforms['query'])
# image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
#                                           data_transforms['gallery'])
# query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
# gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)

# class_names = image_datasets['query'].classes


# ###  Extract Features

# def extract_feature(dataloaders):
    
#     features =  torch.FloatTensor()
#     count = 0
#     idx = 0
#     for data in tqdm(dataloaders):
#         img, pid, camid, camids, target_view, imgpath=data
#         # img, label = data
#         # Uncomment if using GPU for inference
#         if torch.cuda.is_available():
#             img, camid, target_view = img.cuda(),camid.cuda(), target_view.cuda()

#         output = model(cam_label=camids, view_label=target_view) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size

#         n, c, h, w = img.size()
        
#         count += n
#         features = torch.cat((features, output.detach().cpu()), 0)
#         idx += 1
#     return features
def extract_feature(dataloaders,num_query):
    
    feats =  []
    pids = []
    camids = []
    count = 0
    idx = 0
    for data in tqdm(dataloaders):
        img, pid, camid, camids, target_view, imgpath=data
        # img, label = data
        # Uncomment if using GPU for inference
        if torch.cuda.is_available():
            img, camid, target_view = img.cuda(),camid.cuda(), target_view.cuda()

        output = model(img,cam_label=camids, view_label=target_view) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size

        # n, c, h, w = img.size()
        
        # count += n
        feats = torch.cat((features, output.detach().cpu()), 0)
        feats.append(output.detach().cpu())
        pids.extend(np.asarray(pid))
        camids.extend(np.asarray(camid))
        idx += 1
    features = torch.cat(feats, dim=0)
    qf = feats[:num_query]
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
        # gallery
    gf = feats[num_query:]
    g_pids = np.asarray(pids[num_query:])

    g_camids = np.asarray(camids[num_query:])
    return qf,q_pids,q_camids,gf,g_pids,g_camids
# Extract Query Features

query_feature,query_label,query_cam,gallery_feature,gallery_label,gallery_cam= extract_feature(val_loader)

# # Extract Gallery Features

# gallery_feature = extract_feature(gallery_loader)

# # Retrieve labels

# gallery_path = [d[-1] for d in image_datasets['gallery']]
# query_path = [d[-1] for d in image_datasets['query']]

# gallery_cam,gallery_label = get_id(gallery_path)
#  = get_id(query_path)


# ## Concat Averaged GELTs

concatenated_query_vectors = []
for query in tqdm(query_feature):
    fnorm = torch.norm(query, p=2, dim=1, keepdim=True)#*np.sqrt(H*W)
    query_norm = query.div(fnorm.expand_as(query))
    concatenated_query_vectors.append(query_norm.view((-1)))

concatenated_gallery_vectors = []
for gallery in tqdm(gallery_feature):
    fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True)#*np.sqrt(H*W)
    gallery_norm = gallery.div(fnorm.expand_as(gallery))
    concatenated_gallery_vectors.append(gallery_norm.view((-1)))
  

# ## Calculate Similarity using FAISS

index = faiss.IndexIDMap(faiss.IndexFlatIP(H*W*D))

index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gallery_label))

def search(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k


# ### Evaluate 

rank1_score = 0
rank5_score = 0
ap = 0
count = 0
for query, label in zip(concatenated_query_vectors, query_label):
    count += 1
    label = label
    output = search(query, k=10)
    rank1_score += rank1(label, output) 
    rank5_score += rank5(label, output)

    print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count-rank1_score), end="\r")
    ap += calc_ap(label, output)

print("Rank1: %.3f, Rank5: %.3f, mAP: %.3f"%(rank1_score/len(query_feature), 
                                             rank5_score/len(query_feature), 
                                             ap/len(query_feature)))    

