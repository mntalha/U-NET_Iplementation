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
        
        self.images_path = os.path.join(root_dir,"CXR_png")
        
        self.masks_path =  os.path.join(root_dir,"masks")
        
        self.tests_path = os.path.join(root_dir,"test")
        
        img_list_row = os.listdir(self.images_path)

        mask_list_row = os.listdir(self.masks_path)
        

  
        for x,i in enumerate(mask_list_row):
            if i[-8:] != "mask.png":
                mask_list_row[x] = mask_list_row[x][:-4]+"_mask.png"  
                
         
        img_list = []
        for x,i in enumerate(img_list_row):
            str = i[:-4] + "_mask.png"
            if str in mask_list_row:
                img_list.append(i)
                
        self.image_list = img_list
        
        self.mask_list = os.listdir(self.masks_path)
        
        
        self.tests_list = os.listdir(self.tests_path)
      
                
        self.transform = transform

        
    def __len__(self):
        return len(self.image_list)  

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name =os.path.join(self.images_path,self.image_list[idx])
        
        img = io.imread(img_name,as_gray=True)
        
        if img.dtype == ('float64'):
            img = img *255.0
            img = img.astype(np.uint8)

        mask_name =  os.path.join(self.masks_path,self.mask_list[idx])       
        
        mask = io.imread(mask_name)
        
        sample = {'image': img, 'mask': mask}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
            sample["mask"] = self.transform(sample["mask"])

        return sample
        
def transform_operation( width = 128, height = 128) :
    import torchvision.transforms as transforms
    transforms_output = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([width,height]),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ])
    return transforms_output
        
if __name__ == "__main__":
    
    train_dataset = CustomDataset(root_dir , transform=(transform_operation(1028,1028)))
    
    trainloader = DataLoader(dataset=train_dataset,batch_size=32,num_workers=12, shuffle=False,pin_memory=True)
    

        
        