import glob
import os
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
import numpy as np
import cv2
import torch
import re

from src.utils.io import path_leaf, load_image

fileExtensions = ["jpg", "jpeg", "png", "tiff"]


class ImagesDataset(Dataset):
    """
    Ce dataset renvoie une image à la fois. Il sera certainement plus pertinent qu'il renvoie à chaque fois une image
    et la phase associée, mais cela prend de réfléchir à la gestion de la phase. Une facon de faire serait de stocker
    cette information dans un fichier qui puisse être lu par pandas. A chaque image
    """
    def __init__(self, groundtruth_list, path_weights, path_img, shape=(512, 512), RNN_len=100, recursive=True): #passer le tableau ou le chemin
        """
        A compléter éventuellement pour prendre en entrée le chemin vers le fichier des phases (groundtruth)
        :param path_img:
        :param shape:
        :param recursive:
        """
        super(ImagesDataset, self).__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.path_img = path_img #adresse du DOSSIER d'images
        self.shape = shape
        self.path_weights = path_weights
        self.da_core = None  # Data augmentation instance. Only initialized if required
        
        self.groundtruth_list = groundtruth_list
        self.RNN_len = RNN_len
        
        self.img_filepath = []

        for file in os.listdir(self.path_img):
            self.img_filepath.extend(glob.glob(self.path_img + file + '/' + '*.pt', recursive=recursive)) #was .pt

        img_filenames = [path_leaf(path).split('.')[0] for path in self.img_filepath] #Liste de toutes les images ['frame0', 'frame1', ...]

        self.img_filepath = np.asarray(self.img_filepath)
        img_argsort = np.argsort(img_filenames)
        self.img_filepath = self.img_filepath[img_argsort] #array de tous les paths (\data01\frameX.jpg), pas dans l'ordre
        self.img_filepath = natsorted(self.img_filepath)
        self.img_filepath = np.array(self.img_filepath)


    def set_data_augmentation_core(self, da_core):
        # self.da_core = da_core
        pass

    def subset(self, indices):
        self.img_filepath = self.img_filepath[indices]

    def __len__(self):
        return len(self.img_filepath)
    
    #diviser et multiplier dans getitem
    # - RNN_len
    #padding

    def __getitem__(self, item):
        """
        Item est un index (entier), le dataset renvoie l'image et la groundtruth correspondante
        :param item:
        :return:
        """
        sequence_img = torch.FloatTensor()
        if len(self.img_filepath[item:]) > self.RNN_len:
            for tensor in self.img_filepath[item : item + self.RNN_len]:
                img = torch.load(tensor, map_location = torch.device('cpu')) #tensor contient à la fois le n° du dossier et le n° de frame 
                sequence_img = torch.cat(sequence_img, img, 0)
            sequence_phase = self.read_phase(self.img_filepath[item : item+self.RNN_len])
        else:
            for tensor in self.img_filepath[item:]:
                img = torch.load(tensor, map_location = torch.device('cpu'))
                sequence_img = torch.cat(sequence_img, img, 0)
            sequence_phase = self.read_phase(self.img_filepath[item:])
        
        seq_len = sequence_img.shape[0]     
        
        return self.pad_seq(sequence_img), self.pad_seq(sequence_phase), seq_len
    
    def pad_seq(self, array):
        shape = array.shape
        dtype = array.dtype
        pad = self.RNN_len - shape[0]
        padding = [(0, pad)] + [(0, 0) for _ in shape[1:]]
        padded_array = np.pad(array, padding, mode='constant', constant_values=-1)

        if dtype==int:
            return padded_array.astype(dtype)
        else:
            return padded_array.astype(np.float32)

    #padding dans getitem
    #padding dans le trainer
    #ignorer les valeurs dans CrossEntropyLoss

    def get_classes_weight(self):
        """ Fonction à implémenter potentiellement: elle charge ou calcul une pondération par classe permettant de les
        équilibrer.
        J'ai mis du pseudo-code à compléter selon le besoin, cela dépend de l'utilisation
        """

        classes_weight_path = os.path.join(self.path_weights, 'classes_weight.npy') # chemin de sauvergarde des weights

        if os.path.exists(classes_weight_path):
            print("Loading existing weights for class balance from", classes_weight_path)
            class_weights = np.load(classes_weight_path)
        else:
            print("Building weights for class balance")
            classes_counts = np.zeros(128,
                                      dtype=int)  # Arbitrary number because the number of classes is unknown at this point
            for i in range(len(self.img_filepath)):
                phase = self.read_phase(self.img_filepath[i])
                u, counts = np.unique(phase, return_counts=True)
                classes_counts[u] += counts
            classes_counts = classes_counts[
                             :np.max(np.nonzero(classes_counts)) + 1]  # Keep only the classes that have a count
            n_classes = len(classes_counts)
            n_samples = classes_counts.sum()
            class_weights = (n_samples / (n_classes * classes_counts + 1e-8)).astype(np.float32)
            np.save(classes_weight_path, class_weights)
            print('Weights stored in ', classes_weight_path)
        return class_weights
    
                        
    def read_phase(self, filepaths):
        Phases = []
        for filepath in filepaths:
            #find the number X of the video and the number Y of the image, saved in a file dataX with the name frameY
            temp = re.findall(r'\d+', filepath)
            res = list(map(int, temp))
            X = res[-2] - 1  #les indices de la list groundtruth démarrent à 0 et les fichiers dataX démarrent à 1
            Y = res[-1]
            groundtruth = self.groundtruth_list[X]
                   
            B = (groundtruth.at[Y,"Frame,Steps"]) #groundtruth est un DataFrame créé par Pandas regroupant toutes les informations Frame,Steps
             
            temp = re.findall(r'\d+', B) 
            res = list(map(int, temp)) #getting numbers from the string B = "frame_number,step_number" 
            
            #if there was no Steps value specified, then there is no surgical phase on the image
            if len(res) == 2:
                Phase = res[1]
            else:
                Phase = 0
                
            Phases.append(Phase)
        
        return torch.LongTensor(Phases)
        