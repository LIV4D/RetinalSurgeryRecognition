import yaml
from src.services.trainer import Trainer

config_path = 'configs/config.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

trainer = Trainer(config)
trainer.train()