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
        self.batch_first = True
        super(MyNetwork_RNN, self).__init__()
        h0 = nn.Parameter(torch.zeros(2, 1, 512)) #was (2, 1, 2) pour les deux lignes
        c0 = nn.Parameter(torch.zeros(2, 1, 512))
        h0 = h0.cuda()
        c0 = c0.cuda()
        self.h0 = h0
        self.c0 = c0
        #paramètre donc des gradients seront calculés dessus
        h0_RNN = nn.Parameter(torch.zeros(1, 1, self.config['CNN']['n_classes'])) #was (1, 1, 5)
        h0_RNN = h0_RNN.cuda()
        self.h0_RNN = h0_RNN
        #pas forcément nécessaire
        
        self.LSTM = nn.LSTM(1024, 512, 2) #was (1024,512,2) #was (19, 10, 2)
        self.RNN = nn.RNN(512, self.config['CNN']['n_classes'], 1) #was ((10, 5, 1))

    def forward(self, input_tensors, seq_size): #récupérer la batch_size, répéter le h0 pour chaque échantillon du batch
        b = input_tensors.size(0) #batch_size
        h0 = self.h0.repeat(1, b, 1)
        c0 = self.c0.repeat(1, b, 1)
        h0_RNN = self.h0_RNN.repeat(1, b, 1)
        
        input_tensors = pack_padded_sequence(input_tensors, lengths = seq_size, batch_first = self.batch_first, enforce_sorted = False)
        intermed, (hn, cn) = self.LSTM(input_tensors,(h0, c0))
        output, hn_RNN = self.RNN(intermed, h0_RNN)
        OUT = self.unpad(output)
        return OUT[0] #retourne un tenseur contenant les sorties du RNN pour chacune des 100 images de la séquence
    
    def unpad(self, x):
        return pad_packed_sequence(x, batch_first=self.batch_first, padding_value = -1, total_length = self.config['Dataset']['rnn_sequence'])[0] #voir doc, selon la dimension de X

#dimensions input_tensors : (seq,batch,num_features)
#dimensions output : seq,batch,num_classes

