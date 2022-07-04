import PIL
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def tokenize(image,patch_size):
    if type(image) == np.ndarray :
        image = np.transpose(image,[2,0,1])
        image = torch.from_numpy(image)
    c,h,w = image.shape
    is_reshape = False
    if w%patch_size != 0:
        w = (w//patch_size + 1)*patch_size
        is_reshape = True
    if h%patch_size !=0:
        h = (h//patch_size + 1)*patch_size
        is_reshape = True

    if is_reshape:
        image = transforms.Resize((h,w),
            interpolation=transforms.InterpolationMode.BILINEAR)(image)
    image = torch.reshape(image,(c,h//patch_size,patch_size,w//patch_size,patch_size))
    image = torch.permute(image,[1,3,2,4,0])
    return image

def read_file(read_func,file_name,transforms):
    image = read_func(file_name)
    image = transforms(image)
    return image
    
class VisionTransDataset(Dataset) :
    def __init__(self,file_path,transforms,read_func) :
        super.__init__(self)
        self.file_path = file_path

    def __len__(self) :
        return len(os.listdir(self.file_path))

    def __getitem__(self,idx):
        file_name = self.file_path[idx]
        return read_file(read_func,file_name,transforms)

def vision_data_loader(dataset,*args):
    return Dataloader(dataset,args)

