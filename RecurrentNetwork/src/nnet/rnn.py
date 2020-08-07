from torch import nn
import torch
import torchvision.models.segmentation as models
import torchvision.models as allmodels
from src.nnet.abstract_network import AbstractNet
from .blocks.core import ConvLayer

class MyNetwork_RNN(AbstractNet):
    """
    Ton CNN qui dérive de AbstractNet (qui dérive de nn.Module). Si jamais tu veux construire ton propre réseau,
    tu seras surement amener à construire des briques, que j'ai placé dans le module blocks
    """
    def __init__(self, config):
        self.config = config
        super(MyNetwork_RNN, self).__init__()
        self.h0 = nn.Parameter(torch.zeros(2, self.config['Dataset']['batch_size'], 512))
        self.c0 = nn.Parameter(torch.zeros(2, self.config['Dataset']['batch_size'], 512))
        #paramètre donc des gradients seront calculés dessus
        self.h0_RNN = nn.Parameter(torch.zeros(1, self.config['Dataset']['batch_size'], self.config['CNN']['n_classes']))
        #pas forcément nécessaire
        
        self.LSTM = nn.LSTM(1024, 512, 2)
        self.RNN = nn.RNN(512, self.config['CNN']['n_classes'], 1)
        
        softmax = nn.Softmax(2)

        self.softmax = softmax

    def forward(self, input_tensors):
        intermed = self.LSTM(input_tensors, self.h0, self.c0)
        output = self.RNN(intermed, self.h0_RNN)
        return self.softmax(output) #was [0]

#dimensions input_tensors : (seq,batch,num_features)
#dimensions output : seq,batch,num_classes

