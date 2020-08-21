import tqdm
import torch
import os
import numpy as np
import random
import re

from src.utils.torch_utils import DataParallel
from src.dataset_CNN.dataset_manager import DatasetManager
from src.nnet.cnn import MyNetwork
from src.utils.io import create_folder

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
        
        if self.device != 'cpu':
            self.network = self.network.cuda(self.device)
        if self.multi_gpu:
            self.network = DataParallel(self.network, device_ids=self.manager_config['gpu'])
        
        self.network.eval()

    def features_from_CNN(self):
        """
        Creation of an array containing features to describe all the images (CNN's output)
        """

        dataloader = self.datasetManager.get_dataloader()
        print("\nFeatures obtention with CNN")
        print("-"*15)
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            img = self.to_device(batch[0])
            img_name = batch[2][0]
            
            temp = re.findall(r'\d+', img_name)
            res = list(map(int, temp))
            X = res[-2]
            Y = res[-1]
            
            savepath = os.path.join(self.output_dir, 'data%i'%X)
            create_folder(savepath)
            
            out_CNN = self.network(img)  
        
            torch.save(out_CNN, os.path.join(savepath,'features_tensor%i.pt'%Y))
        
        
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
    
        
    
    

