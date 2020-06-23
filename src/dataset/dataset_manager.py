import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader

from .dataset import ImagesDataset


class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.img_size = self.config['img_size']
        self.groundtruth_path = self.config['groundtruth_path']
        self.dataset_args = {'path_img': self.config['img_folder'], #bouclage sur les dossiers ???
                             'shape': self.img_size,
                             'recursive': self.config['load_recursirvely']}
        self.groundtruth_list = self.get_ground_truth_list(self.groundtruth_path)
        self.dataset = ImagesDataset(self.groundtruth_list, **self.dataset_args) #on passe le tableau groundtruth directement dans la classe ImagesDataset
        print('Found %i images for current experimentation' % len(self.dataset))
        """
        The validation set will only be initialed if build_validation_set is called (for training, not testing)
        """
        self.validation_dataset = None
        

    def build_validation_set(self):
        len_dataset = len(self.dataset)
        indices = np.arange(len_dataset)
        np.random.shuffle(indices)
        valid_len = int(len_dataset * self.config['validation_ratio'])
        valid_indices = indices[:valid_len]
        train_indices = indices[valid_len:]
        self.dataset.subset(train_indices)
        self.validation_dataset = ImagesDataset(self.groundtruth_list, **self.dataset_args)
        self.validation_dataset.subset(valid_indices)


    def get_dataloader(self, shuffle=True, drop_last=True):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          pin_memory=self.config['pin_memory'],
                          drop_last=drop_last,
                          num_workers=self.config['num_workers'])

    def get_classes_weights(self, tensor=True):
        classes_weight = self.dataset.get_classes_weight()
        if tensor:
            return torch.from_numpy(classes_weight)
        else:
            return classes_weight

    def get_validation_dataloader(self):
        if self.validation_dataset is not None:
            return DataLoader(self.validation_dataset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              pin_memory=self.config['pin_memory'],
                              drop_last=True,
                              num_workers=self.config['num_workers'])
        else:
            raise ValueError('The validation dataset has not been created!')
            
    def get_ground_truth_list(self,groundtruth_path): #donner l'adresse du fichier contenant .csv contenant les Frames,Steps
        dirList = os.listdir(groundtruth_path)
        groundtruth_list = []
        for dir in dirList:
            A = pd.read_csv(groundtruth_path + '\\' + dir, sep = '[\t;]')
            groundtruth_list.append(A)
        return groundtruth_list #renvoie une liste contenant les DataFrame 
    
        
        
