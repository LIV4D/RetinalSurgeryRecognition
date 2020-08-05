from torch import nn
import torchvision.models.segmentation as models
import torchvision.models as allmodels
from src.nnet.abstract_network import AbstractNet
from .blocks.core import ConvLayer

def _get_network(network):
    return {
        "deeplabv3_resnet101": models.deeplabv3_resnet101,
        "fcn_resnet101": models.fcn_resnet101,
        "inception": allmodels.inception_v3,
        "resnet": allmodels.resnet50,
    }[network]
# Il faudra éventuellement remplacer ces modèles par des modèles de classification (ici, ce sont des modèles de segmen-
# tation.


class MyNetwork(AbstractNet):
    """
    Ton CNN qui dérive de AbstractNet (qui dérive de nn.Module). Si jamais tu veux construire ton propre réseau,
    tu seras surement amener à construire des briques, que j'ai placé dans le module blocks
    """
    def __init__(self, config):
        self.config = config
        super(MyNetwork, self).__init__()

        self.network = _get_network(self.config['model'])(pretrained=self.config['pretrained'])
#        self.network.aux_logits = None

        if not self.config['continue_training']:
            # Quand on fait du transfer learning, on peut choisir de poursuivre l'entrainement de toutes les couches
            # ou uniquement de celles qui ont été modifiées pour la nouvelle tâche.
            for p in self.network.backbone.parameters():
                p.requires_grad = False

        # Un exemple d'utilisation d'une brique concue préalablement et réutilisable ailleurs dans le code.
        """fcn = nn.Sequential(ConvLayer(1024, 256, kernel_size=(3, 3), dropout=0.1, activation='relu', norm='batch'),
                            nn.Conv2d(256, self.config['n_classes'], kernel_size=(1, 1), stride=(1, 1)))"""     
        
        fcn = nn.Sequential(nn.Linear(2048, 1024, bias = True),nn.Linear(1000, self.config['n_classes'], bias = True))
        self.network.fc = fcn

    def forward(self, input_tensors):
        return self.network(input_tensors)[0]
    