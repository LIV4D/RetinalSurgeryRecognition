import random
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from src.utils.torch_utils import DataParallel
from src.dataset.dataset_manager import DatasetManager
from src.nnet.cnn import MyNetwork

class Manager:
    def __init__(self, config):
        self.config = config
        self.manager_config = self.config['Manager']
        self.network = MyNetwork(self.config['CNN'])

        self.datasetManager = DatasetManager(self.config['Dataset'])

        self.multi_gpu = False  # Initialized in setup_gpu()
        self.device = 'gpu'  # Initialized in setup_gpu()
        self.set_seed() # Fix la seed, pour assurer la reproducibilité des expériences
        self.setup_gpus()
        self.exp_path = os.path.join(self.manager_config['save_point'], self.manager_config['experiment_name'])
        self.tb_writer = SummaryWriter(os.path.join(self.exp_path, 'tensorboard/')) # On utilise tensorBoard pour le suivi des expériences
        self.network.savepoint = os.path.join(self.exp_path, 'trained_model') # Point de sauvegarde du réseau
        self.softmax = torch.nn.Softmax(dim=1)
        if self.device != 'cpu':
            self.network = self.network.cuda(self.device)
        if self.multi_gpu:
            self.network = DataParallel(self.network, device_ids=self.manager_config['gpu'])

    def set_seed(self):
        seed = self.manager_config['seed']
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_gpus(self):
        gpu = self.manager_config['gpu']
        if gpu != 'none':
            if not isinstance(gpu, list):
                gpu = self.manager_config['gpu'] = [gpu]
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
