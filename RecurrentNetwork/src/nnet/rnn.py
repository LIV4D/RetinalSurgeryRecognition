from torch import nn
import torch
from src.nnet.abstract_network import AbstractNet
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

class MyNetwork_RNN(AbstractNet):
    """
    Ton CNN qui dérive de AbstractNet (qui dérive de nn.Module). Si jamais tu veux construire ton propre réseau,
    tu seras surement amener à construire des briques, que j'ai placé dans le module blocks
    """
    def __init__(self, config):
        self.config = config
        super(MyNetwork_RNN, self).__init__()
        self.h0 = nn.Parameter(torch.zeros(2, 1, 512))
        self.c0 = nn.Parameter(torch.zeros(2, 1, 512))
        #paramètre donc des gradients seront calculés dessus
        self.h0_RNN = nn.Parameter(torch.zeros(1, 1, self.config['CNN']['n_classes']))
        #pas forcément nécessaire
        
        self.LSTM = nn.LSTM(1024, 512, 2)
        self.RNN = nn.RNN(512, self.config['CNN']['n_classes'], 1)

    def forward(self, input_tensors, seq_size): #récupérer la batch_size, répéter le h0 pour chaque échantillon du batch
        b = input_tensors.size(0)
        h0 = self.h0.repeat(1, b, 1)
        c0 = self.c0.repeat(1, b, 1)
        h0_RNN = self.h0_RNN.repeat(1, b, 1)
        
        input_tensors = pack_padded_sequence(input_tensors, seq_size, batch_first = self.batch_first, enforce_sorted = False)
        intermed = self.LSTM(input_tensors, h0, c0)
        output = self.RNN(intermed, h0_RNN)
        OUT = self.unpad(output)
        return OUT #was [0]
    
    def unpad(self, x):
        return pad_packed_sequence(x, batch_first=self.batch_first)[0] #voir doc, selon la dimension de X

#dimensions input_tensors : (seq,batch,num_features)
#dimensions output : seq,batch,num_classes

