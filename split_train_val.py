import os
import pandas as pd
import re
import numpy as np
import glob
import ntpath
from shutil import copyfile
from src2.utils.io import path_leaf, create_folder

def read_phase(groundtruth_list, filepath):
    #find the number X of the video and the number Y of the image, saved in a file dataX with the name frameY
    temp = re.findall(r'\d+', filepath)
    res = list(map(int, temp))
    X = res[-2] - 1  #les indices de la list groundtruth démarrent à 0 et les fichiers dataX démarrent à 1
    Y = res[-1]
    groundtruth = groundtruth_list[X]      
    B = (groundtruth.at[Y,"Frame,Steps"]) #groundtruth est un DataFrame créé par Pandas regroupant toutes les informations Frame,Steps
     
    temp = re.findall(r'\d+', B) 
    res = list(map(int, temp)) #getting numbers from the string B = "frame_number,step_number" 
    
    #if there was no Steps value specified, then there is no surgical phase on the image
    if len(res) == 2:
        Phase = res[1]
    else:
        Phase = 0
    
    return Phase


groundtruth_path = '/home/clement/CATARACTS/GroundTruth/'
train_folder = '/home/clement/CATARACTS/PytorchData/train/'
#groundtruth_path = 'C:\\Users\lucas\OneDrive\Documents\Professionnel\StageMontreal2020\Python\GroundTruth\\'
#train_folder = 'C:\\Users\lucas\OneDrive\Documents\Professionnel\StageMontreal2020\Python\Training_Data\\'

val_folder = '/home/clement/CATARACTS/PytorchData/val/'
#val_folder = 'C:\\Users\lucas\OneDrive\Documents\Professionnel\StageMontreal2020\Python\\NEWTEST\\'


dirList = os.listdir(groundtruth_path)
dirList.sort()
groundtruth_list = []
for dir in dirList:
    A = pd.read_csv(os.path.join(groundtruth_path, dir), sep = '[\t;]', engine = 'python')
    groundtruth_list.append(A)
    
img_filepath = []
for file in os.listdir(train_folder):
    img_filepath.extend(glob.glob(train_folder + file + '/' + '*.jpg', recursive=True))
        
img_filenames = [path for path in img_filepath] #Liste de toutes les images ['frame0', 'frame1', ...]


len_dataset = len(img_filenames)
indices = np.arange(len_dataset)
np.random.shuffle(indices)
indices = indices[:int(len_dataset*0.2)]

for i in indices:
    path = img_filenames[i]
    Phase = read_phase(groundtruth_list, path)
    new_folder = val_folder + 'Phase' + str(Phase)
    create_folder(new_folder)
    head, num_img = ntpath.split(path)
    direct, num_video = ntpath.split(head)
    tail = num_video + '_' + num_img
    destination = new_folder + '/' + tail 
    os.rename(path, destination)



