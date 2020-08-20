from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import re
import numpy as np

def read_phase(groundtruth, img):        
    B = (groundtruth.at[img,"Frame,Steps"]) #groundtruth est un DataFrame créé par Pandas regroupant toutes les informations Frame,Steps
    
    temp = re.findall(r'\d+', B) 
    res = list(map(int, temp)) #getting numbers from the string B = "frame_number,step_number" 
    
    #if there was no Steps value specified, then there is no surgical phase on the image
    if len(res) == 2:
        Phase = res[1]
    else:
        Phase = 0
    
    return Phase

groundtruth_path = '/home/clement/CATARACTS/GroundTruth/'
#groundtruth_path = 'C:\\Users\lucas\OneDrive\Documents\Professionnel\StageMontreal2020\Python\GroundTruth'
savepath = '/home/clement/Documents/Lucas/RetinalSurgeryRecognition/Class_Repartition/'
#savepath = 'C:\\Users\lucas\OneDrive\Documents\Professionnel\StageMontreal2020\Python\Class_repartition'


tb_writer = SummaryWriter(savepath)

dirList = os.listdir(groundtruth_path)
dirList.sort()
groundtruth_list = []
for dir in dirList:
    A = pd.read_csv(os.path.join(groundtruth_path, dir), sep = '[\t;]', engine = 'python')
    groundtruth_list.append(A)


Classes = np.zeros(19, dtype = int)
for groundtruth in groundtruth_list:
    L = len(groundtruth)
    for img in range(L):
        Phase = read_phase(groundtruth, img)
        Classes[Phase] += 1
    
for i in range(len(Classes)):
    tb_writer.add_scalar('Répartition des classes', Classes[i], i)
