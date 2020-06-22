import yaml
from src.services.trainer import Trainer

config_path = 'src/configs/configFile.yaml'
output_dir = 'src/trained'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

trainer = Trainer(config,output_dir)
trainer.train()