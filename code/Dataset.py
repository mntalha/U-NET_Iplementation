# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:59:09 2021

@author: talha
"""

import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


root_dir = "../data" 


class CustomDataset(Dataset):
    # X-RAY MASK DATASET 
    
    def __init__(self,root_dir, transform = None):
        
        self.root_dir = root_dir
        
        images_path = os.path.join(root_dir,"CXR_png")
        
        masks_path =  os.path.join(root_dir,"masks")
        
        tests_path = os.path.join(root_dir,"test")
        
        img_list_row = os.listdir(images_path)

        mask_list_row = os.listdir(masks_path)

  
        for x,i in enumerate(mask_list_row):
            if i[-8:] != "mask.png":
                mask_list_row[x] = mask_list_row[x][:-4]+"_mask.png"  
                
         
        img_list = []
        for x,i in enumerate(img_list_row):
            str = i[:-4] + "_mask.png"
            if str in mask_list_row:
                img_list.append(i)
                
        self.image_list = img_list
        
        self.mask_list = mask_list_row
        
        tests_list = os.listdir(tests_path)
        
        self.tests_list =tests_list
               
        self.transform = transform

        
    def __len__(self):
        return len(self.image_list)  

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name =