import yaml
from src2.services.tester import Tester

config_path = 'src2/configs/configFile.yaml'

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

tester = Tester(config, '1')
tester.inference()


