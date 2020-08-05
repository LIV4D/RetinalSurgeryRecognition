import tqdm
import torch
import torch.nn as nn
import os
import yaml
import numpy as np

import sklearn.metrics
from src.services.abstract_manager import Manager


class Trainer(Manager):
    def __init__(self, config_file, output_dir):
        super(Trainer, self).__init__(config_file)
        self.train_config = self.config['Training']
        self.output_dir = output_dir
        self.datasetManager.build_validation_set()

        self.loss = []
        self.setup_loss()
        #self.setup_optims()
        self.is_first_valid = True
        self.best_valid_loss = 1e8

    def on_epoch(self, e):
        """
        Sub-training loop (inside an epoch). Iterate over the whole dataloader once.
        :param e: Current epoch
        :return:
        """

        dataloader = self.datasetManager.get_dataloader()
        length_dataloader = len(dataloader)
        print("Epoch %i"%e)
        print("-"*15)
        out_cat = torch.FloatTensor()
        gts_cat = torch.LongTensor()
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            index = e*length_dataloader+i
            batch = self.to_device(batch)
            img = batch[0]
            gts = batch[1]
            
            out_CNN = self.network(img)  
            
            out_cat = torch.cat((out_cat,out_CNN),0)
            gts_cat = torch.cat((gts_cat,gts),0)
            
            if index % self.train_config['rnn_sequence'] == 0:
                out_RNN = self.LSTM(out_cat)
                loss = self.loss(out_RNN, gts_cat)
    
                if index % self.config['Validation']['validation_step'] == 0:
                     """
                     Validation and saving of the model
                     """
                        
                     with torch.no_grad():
                         valid_loss = self.validate(index)
                         if valid_loss < self.best_valid_loss:
                             self.best_valid_loss = valid_loss
                             filename = 'trained_model_iter_%i_loss_%.4f.pth'%(index, valid_loss)
                             filename = os.path.join(self.output_dir, 'trained_model', filename)
                             self.network.save_model(filename, optimizers=self.opt)
                
                self.backward_and_step(loss) #On appel la backpropagation

    def train(self):
        """
        Loop over the number of epochs (one epoch = one complete look of the training data).
        :return:
        """
        for e in range(self.train_config['nb_epochs']):
            self.on_epoch(e)

        with open(os.path.join(self.exp_path, 'config.yml'), 'w') as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def validate(self, current_index):
        """
        Load the validation set, infer on it, display (input, groundtruth, prediction) on tensorBoard and compute
        classification metric
        :param current_index: Current training index=index_current_epoch*nb_batch_per_epoch+index_current_batch
        :return: The average validation loss (scalar)
        """    
        loss_out = []
        n_class = self.config['CNN']['n_classes']
        gts_cat = torch.LongTensor()
        pred_cat = torch.LongTensor()
        out_cat = torch.FloatTensor()
        Validation = self.datasetManager.get_validation_dataloader()
        length = len(Validation)
        for i, batch in enumerate(Validation):
            print('Batch %i out of %i'%(i,length))
            index = current_index*length+i
            batch = self.to_device(batch)
            img = batch[0]
            gts = batch[1]
            out = self.network(img)
            out = self.softmax(out)
            loss = self.loss(out,gts)
            pred = torch.argmax(out, 1, keepdim = True)
            pred = pred.view(-1)
            loss_out.append(loss.item())
            
            gts_cat = torch.cat((gts_cat,gts.cpu()),0)
            pred_cat = torch.cat((pred_cat,pred.cpu()),0)
            out_cat = torch.cat((out_cat,out.cpu()),0)
            
            
        gts_cat = gts_cat.numpy()
        pred_cat = pred_cat.numpy()

        gts_onehot = self.one_hot(gts_cat, n_class)
        pred_onehot = self.one_hot(pred_cat, n_class)
        proba = out_cat.numpy()
        
        fpr = dict()
        tpr = dict()
        AUC_roc = dict()
        Mean_roc = []
        for k in range(n_class):
            fpr[k], tpr[k], _ = sklearn.metrics.roc_curve(gts_onehot[:,k], proba[:,k])
            AUC_roc[k] = sklearn.metrics.auc(fpr[k], tpr[k])        
            self.tb_writer.add_scalar('AUC_Roc classe %i'%k, AUC_roc[k], current_index)
            
            Mean_roc.append(AUC_roc[k])
            
        Accuracy = sklearn.metrics.accuracy_score(gts_cat,pred_cat)  
        
        self.tb_writer.add_scalar('Mean AUC_roc', np.nanmean(Mean_roc), current_index)
        self.tb_writer.add_scalar('Accuracy', Accuracy, current_index)
        self.tb_writer.add_scalar('Loss', np.mean(loss_out), current_index)
        
        return np.mean(loss_out)


    def setup_optims(self):
        """
        Configure the training solver according to the config file. Use adam solver.
        :return:
        """
        lr = self.train_config['lr']
        b1 = self.train_config['b1']
        b2 = self.train_config['b2']
        weight_decay = self.train_config['weight_decay']
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr, betas=(b1, b2),
                                    weight_decay=weight_decay)

    def setup_loss(self):
        """
        Configure the training loss according to the config file. The resulting loss is the sum
        of the cross-entropy and/or the dice coefficient. The cross-entropy might be weighted according to the dis-
        tribution of the classes on the training set.
        :return:
        """
        #self.loss = nn.CrossEntropyLoss(weight = self.to_device(self.datasetManager.class_weights))
        self.loss = nn.CrossEntropyLoss()

    def backward_and_step(self, loss):
        """
        Backward propagation and one solver step.
        :param loss:
        :return:
        """
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
    def one_hot(self, gts, n_class):
        L = len(gts)        
        tab = np.zeros((L,n_class))
        for i in range(L):
            tab[i,gts[i]] = 1
        
        return(tab)
    
    def proba_form(self, proba, n_class):
        L = len(proba)        
        tab = np.zeros((L,n_class))
        for i in range(L):
            tab[i, :] = proba[i]
        
        return(tab)        

