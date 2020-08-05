import tqdm
import torch
import os

from src.dataset.dataset_manager import DatasetManager
from src.nnet.cnn import MyNetwork

class Builder:
    def __init__(self, config_file, output_dir):
        self.config = config_file
        self.output_dir = output_dir
        self.datasetManager = DatasetManager(self.config['Dataset'])
        self.network = MyNetwork(self.config['CNN'])

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
            print("%i out of %i"%(i,length_dataloader))
            #batch = self.to_device(batch)
            img = batch[0]
            gts = batch[1]
            
            out_CNN = self.network(img)  
            
            out_cat = torch.cat((out_cat,out_CNN),0)
            gts_cat = torch.cat((gts_cat,gts),0)
        
        torch.save(out_cat, os.path.join(self.output_dir,'features_tensor.pt'))
        torch.save(gts_cat, os.path.join(self.output_dir,'groundtruth_tensor.pt'))
        
    
    

