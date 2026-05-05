import yaml
import os

def load_config(path=None):
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "..", "config", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)