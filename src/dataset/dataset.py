import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch

from utils.io import path_leaf, load_image

fileExtensions = ["jpg", "jpeg", "png", "tiff"]


class ImagesDataset(Dataset):
    """
    Ce dataset renvoie une image à la fois. Il sera certainement plus pertinent qu'il renvoie à chaque fois une image
    et la phase associée, mais cela prend de réfléchir à la gestion de la phase. Une facon de faire serait de stocker
    cette information dans un fichier qui puisse être lu par pandas. A chaque image
    """
    def __init__(self, path_img, shape=(512, 512), recursive=True):
        """
        A compléter éventuellement pour prendre en entrée le chemin vers le fichier des phases (groundtruth)
        :param path_img:
        :param shape:
        :param recursive:
        """
        super(ImagesDataset, self).__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        self.path_img = path_img
        self.shape = shape
        self.da_core = None  # Data augmentation instance. Only initialized if required

        self.img_filepath = []
        self.mask_filepath = []

        for extension in fileExtensions:
            self.img_filepath.extend(glob.glob(self.path_img + "*." + extension, recursive=recursive))

        img_filenames = [path_leaf(path).split('.')[0] for path in self.img_filepath]

        self.img_filepath = np.asarray(self.img_filepath)
        img_argsort = np.argsort(img_filenames)
        self.img_filepath = self.img_filepath[img_argsort]

        """
        Valeurs de normalisation à adopter si tu utilises un modèle de torchvision
        """
        self.normalize_mean = np.asarray([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis].astype(np.float32)
        self.normalize_std = np.asarray([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis].astype(np.float32)

    def set_data_augmentation_core(self, da_core):
        # self.da_core = da_core
        pass

    def subset(self, indices):
        self.img_filepath = self.img_filepath[indices]

    def __len__(self):
        return len(self.img_filepath)

    def normalize(self, img):
        """
        Fonction de normalisation (voir documentation de torchvision)
        :param img:
        :return:
        """
        img = (img.astype(np.float32) - self.normalize_std) / self.normalize_mean
        return img

    def __getitem__(self, item):
        """
        Item est un index (entier), le dataset renvoie l'image est la groundtruth correspondante
        :param item:
        :return:
        """
        img = load_image(self.img_filepath[item])
        img = cv2.resize(img, dsize=self.shape).astype(np.uint8)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.
        img = self.normalize(img)

        return torch.from_numpy(img)
        """
        Il pourrait être pratique que la fonction renvoie à la fois l'image et la vérité terrain,
        return torch.from_numpy(img), torch.from_numpy(phase)
        La lecture de la variable correspond à l'image reste à implémenter
        """



    def get_classes_weight(self):
        """ Fonction à implémenter potentiellement: elle charge ou calcul une pondération par classe permettant de les
        équilibrer.
        J'ai mis du pseudo-code à compléter selon le besoin, cela dépend de l'utilisation
        """

        classes_weight_path = os.path.join(self.path_masks, 'classes_weight.npy') # chemin de sauvergarde des

        if os.path.exists(classes_weight_path):
            print("Loading existing weights for class balance from", classes_weight_path)
            class_weights = np.load(classes_weight_path)
        else:
            print("Building weights for class balance")
            classes_counts = np.zeros(128,
                                      dtype=int)  # Arbitrary number because the number of classes is unknown at this point
            for i in range(len(self.img_filepath)):
                mask = self.get_mask(i)
                u, counts = np.unique(mask, return_counts=True)
                classes_counts[u] += counts
            classes_counts = classes_counts[
                             :np.max(np.nonzero(classes_counts)) + 1]  # Keep only the classes that have a count
            n_classes = len(classes_counts)
            n_samples = classes_counts.sum()
            class_weights = (n_samples / (n_classes * classes_counts + 1e-8)).astype(np.float32)
            np.save(classes_weight_path, class_weights)
            print('Weights stored in ', classes_weight_path)
        return class_weights