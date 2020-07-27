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

for i in range(L):
    if matrix[i][1][1]+matrix[i][1][0] == 0:
        sensitivity.update({'Classe %i'%i : 0})
    else:
        sensitivity.update({'Classe %i'%i : matrix[i][1][1]/(matrix[i][1][1]+matrix[i][1][0])})
        
    if matrix[i][0][1]+matrix[i][0][0] == 0:
        specificity.update({'Classe %i'%i : 0})
    else:
        specificity.update({'Classe %i'%i : 1 - matrix[i][0][1]/(matrix[i][0][1]+matrix[i][0][0])})
        
    accuracy.update({'Classe %i'%i : (matrix[i][1][1]+matrix[i][0][0])/(matrix[i][1][1]+matrix[i][0][0]+matrix[i][1][0]+matrix[i][0][1])})
        
Metrics_writer = SummaryWriter(folder_path)
Metrics_writer.add_scalars('Sensitivity', sensitivity)
Metrics_writer.add_scalars('Specificity', specificity)
Metrics_writer.add_scalars('Accuracy', accuracy)


