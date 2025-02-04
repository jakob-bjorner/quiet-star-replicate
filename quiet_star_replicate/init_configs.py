# find all the configs, and import them, so that the config store will be run in all of them.
from glob import glob
import os


def init_configs():
    # expect this file to be at the root of the library directory, and then we can do relative imports assuming that this module is installed with pip install -e .
    # also make the assumption that the root directory we are in is the name of the package.
    root_path = os.path.dirname(__file__)
    
    # grab all the files ending in config.py
    for config_path in glob(os.path.join(root_path, "**", "config.py")):
        config_import_path = config_path.removeprefix(root_path)[1:] # remove the last slash "/**/config.py"
        base_repo = root_path.split('/')[-1]
        qualified_import = [base_repo] + config_import_path.split('/')
        qualified_import[-1] = "config" # instead of config.py
        exec(f"import {'.'.join(qualified_import)}")

