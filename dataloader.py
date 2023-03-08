import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import datasets as datasets

ML_Datasets = ["SewerML"]

class SewerMLDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader):
        super(SewerMLDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename"])
        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(self.labels[index, :], dtype=torch.float)

        return img, target, path

def get_dataset(dataset_name, ann_root, data_root, split, transform):

    if dataset_name == "SewerML":
        dataset = SewerMLDataset(ann_root, data_root, split=split, transform=transform)
    else:
        raise ValueError("There are no Dataset for the supplied dataset: {}".format(dataset_name))
    
    return dataset
        
def get_dataloader(dataset_name, batch_size, workers, ann_root, data_root, split, transform):

    dataset = get_dataset(dataset_name, ann_root, data_root, split, transform)   
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers = workers, pin_memory=True)
    
    return dataloader, dataset.LabelNames