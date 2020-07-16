import yaml
import shutil
import os
import sys
from src.utils.io import create_folder
from src.services.trainer import Trainer

#changer experiment_name dans le fichier config
config_path = 'src\configs\configFile.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
output_dir = config['Manager']['save_point'] + config['Manager']['experiment_name']

if os.path.exists(output_dir):
    answer = input("Overwrite the experiment ? [y/n]\n")
    if answer == 'y':
        shutil.rmtree(output_dir, ignore_errors = True)
        create_folder(output_dir)
    elif answer == 'n':
        sys.exit("Experiment already done")
    else:
        print('Enter y or n')
else:
    create_folder(output_dir)
        
create_folder(output_dir) 

trainer = Trainer(config,output_dir)
trainer.train()
