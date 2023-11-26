import torch
from torch.utils.data import Dataset
from torchvision import datasets
from data.preprocessor import wordPreproccessor

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):#removed the labels
        self.data = datasets.ImageFolder(root=data, transform=transform)
        # self.labels=labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx] 
        
        customLabel=label+1
        # label = self.labels[idx]
        # print(customLabel)
        return data,customLabel
            
       
        
