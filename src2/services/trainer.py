import tqdm
import torch
import torch.nn as nn
import os
import yaml
import numpy as np
import sklearn.metrics

from src2.services.abstract_manager import Manager


class Trainer(Manager):
    def __init__(self, config_file, output_dir):
        super(Trainer, self).__init__(config_file)
        self.train_config = self.config['Training']
        self.output_dir = output_dir
        self.datasetManager.build_validation_set()

        self.loss = []
        self.setup_loss()
        self.setup_optims()
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
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            index = e*length_dataloader+i
            batch = self.to_device(batch)
            img = batch[0]
            gts = batch[1]

            #add filter here !

            out = self.network(img)
            loss = self.loss(out, gts)
            
            self.tb_writer.add_scalar('Training Loss', loss, index)

            if index % self.config['Validation']['validation_step'] == 0:
                """
                Validation and saving of the model
                """
                # self.tb_writer.add_scalar('Training loss', loss.item(), index)
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
        gts_cat = torch.LongTensor()
        pred_cat = torch.LongTensor()
        Validation = self.datasetManager.get_validation_dataloader()
        length = len(Validation)
        print('\nValidation : %i steps'%length)
        for i, batch in tqdm.tqdm(enumerate(Validation)):
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

        f1_score = sklearn.metrics.f1_score(gts_cat,pred_cat, average = 'macro')
        Kappa = sklearn.metrics.cohen_kappa_score(gts_cat,pred_cat)
        Accuracy = sklearn.metrics.accuracy_score(gts_cat,pred_cat)  
        
        self.tb_writer.add_scalar("f1 score",f1_score,current_index)
        self.tb_writer.add_scalar('Kappa score',Kappa,current_index)
        self.tb_writer.add_scalar('Accuracy', Accuracy, current_index)
        self.tb_writer.add_scalar('Validation Loss', np.mean(loss_out), current_index)
        
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
        self.loss = nn.CrossEntropyLoss(weight = self.to_device(self.datasetManager.class_weights))
        #self.loss = nn.CrossEntropyLoss()

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

