import yaml

with open("../configs/config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.Loader)