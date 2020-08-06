import tqdm
import torch
import os
import numpy as np
import random

from src.utils.torch_utils import DataParallel
from src.dataset.dataset_manager import DatasetManager
from src.nnet.cnn import MyNetwork

class Builder:
    def __init__(self, config_file, output_dir):
        self.config = config_file
        self.output_dir = output_dir
        self.datasetManager = DatasetManager(self.config['Dataset'])
        self.network = MyNetwork(self.config['CNN'])
        
        self.multi_gpu = False  # Initialized in setup_gpu()
        self.device = 'cpu'  # Initialized in setup_gpu()
        self.setup_gpus()
        self.set_seed()
        self.setup_optims()
        
        if self.device != 'cpu':
            self.network = self.network.cuda(self.device)
        if self.multi_gpu:
            self.network = DataParallel(self.network, device_ids=self.manager_config['gpu'])

    def features_from_CNN(self):
        """
        Creation of an array containing features to describe all the images (CNN's output)
        """

        dataloader = self.datasetManager.get_dataloader()
        length_dataloader = len(dataloader)
        print("\nFeatures obtention with CNN")
        print("-"*15)
        out_cat = torch.FloatTensor()
        gts_cat = torch.LongTensor()
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            batch = self.to_device(batch)
            img = batch[0]
            gts = batch[1]
            
            out_CNN = self.network(img)  
            
            out_cat = torch.cat((out_cat,out_CNN.cpu()),0)
            gts_cat = torch.cat((gts_cat,gts.cpu()),0)
        
        torch.save(out_cat, os.path.join(self.output_dir,'features_tensor.pt'))
        torch.save(gts_cat, os.path.join(self.output_dir,'groundtruth_tensor.pt'))
        
        
    def setup_gpus(self):
        gpu = self.config['Manager']['gpu']
        if gpu != 'cpu':
            if not isinstance(gpu, list):
                gpu = self.config['Manager']['gpu'] = [gpu]
            self.multi_gpu = len(gpu) > 1
            device_ids = ','.join([str(_) for _ in gpu])
            self.device = 'cuda:' + device_ids
        print('Using devices:', self.device)

    def to_device(self, tensors):
        if not isinstance(tensors, list):
            tensors = [tensors]
        d_tensor = []
        for t in tensors:
            d_tensor.append(t.to(self.device))
        if len(d_tensor) > 1:
            return d_tensor
        else:
            return d_tensor[0]
        
    def set_seed(self):
        seed = self.config['Manager']['seed']
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    def setup_optims(self):
        """
        Configure the training solver according to the config file. Use adam solver.
        :return:
        """
        lr = self.config['Training']['lr']
        b1 = self.config['Training']['b1']
        b2 = self.config['Training']['b2']
        weight_decay = self.config['Training']['weight_decay']
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(b1, b2),
                                    weight_decay=weight_decay)
        
    
    
