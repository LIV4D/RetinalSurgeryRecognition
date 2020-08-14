import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .dataset import ImagesDataset


class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.img_size = self.config['img_size']
        self.groundtruth_path = self.config['groundtruth_path']
        self.dataset_args = {'path_img': self.config['img_folder'],
                             'shape': self.img_size,
                             'recursive': self.config['load_recursirvely']}
        self.groundtruth_list = self.get_ground_truth_list(self.groundtruth_path)
        self.dataset = ImagesDataset(self.groundtruth_list, self.config['path_weights'], **self.dataset_args) #on passe le tableau groundtruth directement dans la classe ImagesDataset
#        self.class_weights = self.get_classes_weights()
        print('Found %i images for current experimentation' % len(self.dataset))
        """
        The validation set will only be initialed if build_validation_set is called (for training, not testing)
        """
        self.validation_dataset = None #self.build_validation_set
        

    def build_validation_set(self):
        len_dataset = len(self.dataset)
        indices = np.arange(len_dataset)
        np.random.shuffle(indices)
        valid_len = int(len_dataset * self.config['validation_ratio'])
        valid_indices = indices[:valid_len]
        train_indices = indices[valid_len:]
        self.dataset.subset(train_indices)
        self.validation_dataset = ImagesDataset(self.groundtruth_list, self.config['path_weights'], **self.dataset_args)
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
        L = len(self.validation_dataset)
        subset_indices = np.random.randint(0,L,int(L/5))
        #choisi aléatoirement L/5 images numérotées de 0 à L
        subset = Subset(self.validation_dataset,subset_indices)
        if self.validation_dataset is not None:
            return DataLoader(subset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              pin_memory=self.config['pin_memory'],
                              drop_last=True,
                              num_workers=self.config['num_workers'])
        else:
            raise ValueError('The validation dataset has not been created!')
            
    def get_ground_truth_list(self,groundtruth_path): #donner l'adresse du fichier contenant .csv contenant les Frames,Steps
        dirList = os.listdir(groundtruth_path)
        dirList.sort()
        groundtruth_list = []
        for dir in dirList:
            A = pd.read_csv(groundtruth_path + dir, sep = '[\t;]', engine = 'python')
            groundtruth_list.append(A)
        return groundtruth_list #renvoie une liste contenant les DataFrame 
    
        
        
