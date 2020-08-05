import yaml
import shutil
import os
import sys
from src.utils.io import create_folder
from src.services.Features_from_CNN import Builder

#changer experiment_name dans le fichier config
config_path = 'src/configs/configFile.yaml'
#config_path = 'src\configs\configFile.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
output_dir = '/home/clement/Documents/Lucas/RetinalSurgeryRecognition/RecurentNetwork/CNN_features'


if os.path.exists(output_dir):
    answer = input("Overwrite features from CNN ? [y/n]\n")
    if answer == 'y':
        shutil.rmtree(output_dir, ignore_errors = True)
        create_folder(output_dir)
    elif answer == 'n':
        sys.exit("Step already done")
    else:
        print('Enter y or n')
        sys.exit("Error")
else:
    create_folder(output_dir)

trainer = Builder(config,output_dir)
trainer.features_from_CNN()