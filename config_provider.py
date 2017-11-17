from easydict import EasyDict
import yaml
import os


def get_config(model_name):
    cfg_file = 'cfgs/{}.yml'.format(model_name)
    if not os.path.exists(cfg_file):
        return None
    with open(cfg_file, 'r') as f:
        cfg = EasyDict(yaml.load(f))
    return cfg
