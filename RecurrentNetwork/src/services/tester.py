import torch
import os
import cv2
import numpy as np
import sklearn.metrics as skm
from src.services.abstract_manager import Manager
from src.utils.io import create_folder
from torch.utils.tensorboard import SummaryWriter
import tqdm


class Tester(Manager):
    def __init__(self, config, custom_test_name=''):
        config['Manager']['gpu'] = 0  # Inference is done on single GPU (bottleneck is on the CPU side anyway)
        config['Dataset']['batch_size'] = config['Testing']['batch_size']
        config['Dataset']['img_folder'] = config['Testing']['img_folder']
        super(Tester, self).__init__(config)
        self.config_testing = self.config['Testing']
        self.results_path = os.path.join(self.exp_path, 'test' + custom_test_name + '/')
        create_folder(self.results_path)

        if not self.config['CNN']['trained_model_path']:
            """
            First check if the user asked for a particular model loading (which would already be done by the Manager).
            If not, load the latest file present in the experiment folder
            """
            self.load_best_model()
            
        self.n_class = self.config['CNN']['n_classes']    
        self.confusion_matrix = 0
        
    def inference(self):
        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(self.datasetManager.get_dataloader(shuffle=False, drop_last=False))):
                img = self.to_device(batch[0])
                gts = self.to_device(batch[1])
                gts = gts.long()
                seq_len = batch[2]
                
                out_RNN = self.network(img, seq_len)
                
                img = img.view(-1,img.size(-1))
                out_RNN = out_RNN.view(-1, out_RNN.size(-1))
                
                probs = self.softmax(out_RNN[:seq_len])
                pred = torch.argmax(probs[-1], keepdim = True)
                pred = pred.view(-1) 
                
                gts = gts.view(-1)
                gts = gts[:seq_len]
                gts = gts[-1]
                gts = gts.view(-1)
                
                batch_number = str(i)                
                self.eval_batch(pred.cpu(), probs, gts.cpu(), batch_number)
                
        np.save(os.path.join(self.results_path, 'confusion_matrix.npy'), self.confusion_matrix) #self.metrics.confusion_matrix
        #self.results_metrics(self.results_path)

    def eval_batch(self, preds, probs, gts, batch_number):  
        if self.config_testing['eval_performance']:    
            self.report_batch_into_confusion_matrix(gts, preds) #self.metrics.report_batch_into_confusion_matrix


    def load_best_model(self):
        self.network.load(path=self.network.savepoint, load_most_recent=True)
    
    
    def report_batch_into_confusion_matrix(self,gts,pred):
        self.confusion_matrix = self.confusion_matrix + skm.confusion_matrix(gts, pred, labels = [i for i in range(0,self.n_class)])
 
    
    def results_metrics(self, save_path):
       
        matrix_path = os.path.join(save_path, 'confusion_matrix.npy')
        matrix = np.load(matrix_path)
        print(matrix)
                
        L = len(matrix)
        Metrics_writer = SummaryWriter(save_path)
        
        for i in range(L):
            if matrix[i][1][1]+matrix[i][1][0] == 0:
                Metrics_writer.add_scalar('Sensitivity', 0, i)
            else:
                Metrics_writer.add_scalar('Sensitivity', matrix[i][1][1]/(matrix[i][1][1]+matrix[i][1][0]), i)
                
            if matrix[i][0][1]+matrix[i][0][0] == 0:
                Metrics_writer.add_scalar('Specificity', 0, i)
            else:
                Metrics_writer.add_scalar('Specificity', 1 - matrix[i][0][1]/(matrix[i][0][1]+matrix[i][0][0]), i)
                
            Metrics_writer.add_scalar('Accuracy', (matrix[i][1][1]+matrix[i][0][0])/(matrix[i][1][1]+matrix[i][0][0]+matrix[i][1][0]+matrix[i][0][1]), i)
       