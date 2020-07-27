import numpy as np
from torch.utils.tensorboard import SummaryWriter

folder_path = input('Folder path :\n')
matrix_path = folder_path + 'confusion_matrix.npy'

matrix = np.load(matrix_path)
print(matrix)
        
L = len(matrix)
sensitivity = {}
specificity = {}
accuracy = {}
Metrics_writer = SummaryWriter(folder_path)

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
        


