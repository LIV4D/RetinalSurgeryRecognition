import yaml
from src.services.tester import Tester

config_path = 'src/configs/configFile.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

tester = Tester(config, '3')
tester.inference()


