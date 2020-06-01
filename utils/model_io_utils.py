import torch
import matplotlib.pyplot as plt
import glob
import gzip
import json
import os
import re
import models as module_arch
from datetime import datetime
from parse_config import ConfigParser
from pathlib import Path


# Load a trained model using its best weights unless otherwise specified
def load_model_config(experiment_name, train_id, logs='./saved'):
    model_path = os.path.join(logs, 'models', experiment_name, train_id)

    # Load config file
    config_file = os.path.join(model_path, "config.json")
    assert os.path.isfile(config_file)
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


# load all models by train id
def load_trained_models(train_ids, logs='./saved', experiment_name=None, config=None, checkpoint_name='model_best'):
    models = {}
    epochs = {}
    for train_id in train_ids:
        models[train_id], epochs[train_id] = load_trained_model(train_id, logs, experiment_name,
                                                                checkpoint_name=checkpoint_name, config=config)

    return models, epochs


def load_trained_model_by_path(checkpoint_path, config):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    loaded_epoch = checkpoint['epoch']

    print('loaded', checkpoint_path, 'from epoch', loaded_epoch)
    # Load model with parameters from config file
    config_parser = ConfigParser(config, dry_run=True)
    model = config_parser.init_obj('arch', module_arch)

    # TODO: WARNING: Leaving some mipmap layer weights unassigned might lead to erroneous
    #  results (maybe they're not set to zero by default)
    # Assign model weights and set to eval (not train) mode
    #model.load_state_dict(checkpoint['state_dict'], strict=(not zero_other_mipmaps))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model, loaded_epoch


def load_trained_model(train_id, logs='./saved', experiment_name=None,
        checkpoint_name='model_best', config=None):
    assert config is not None or experiment_name is not None, 'Must provide a config file or experiment name'

    if config is None:
        config = load_model_config(experiment_name, train_id, logs)
    
    if experiment_name is None:
        experiment_name = config['name']
    elif experiment_name != config['name']:
        print('Warning: Provided experiment name', experiment_name,
                'differs from config file experiment name', config['name'],
                'using provided experiment name')

    # Load model weights
    model_path = os.path.join(logs, 'models', experiment_name, train_id)
    checkpoint_path = os.path.join(model_path, checkpoint_name) + '.pth'

    if not os.path.isfile(checkpoint_path):
        pose_files = glob.glob(os.path.join(model_path, '*.pth'))

        def extract_epoch(f):
            s = re.findall("checkpoint-epoch(\d+)", f)
            return (int(s[0]) if s else -1, f)

        checkpoint_path = max(pose_files, key=extract_epoch)

    return load_trained_model_by_path(checkpoint_path, config)


# Cereate a libtorch script file containing the model that can be loaded into C++
def create_libtorch_script(train_id, logs, experiment_name, checkpoint_name='model_best',
        model_script_folder='./libtorch-models'):
    model, loaded_epoch = load_trained_model(train_id, logs=logs, checkpoint_name=checkpoint_name,
            experiment_name=experiment_name)
    _create_libtorch_script_from_model(model, train_id, experiment_name, checkpoint_name)


def _create_libtorch_script_from_model(model, epochs, train_id, experiment_name, checkpoint_name,
        model_script_folder):
    sm = torch.jit.script(model)
    model_script_name = '{}-{}-{}-{}-{}.pt'.format(experiment_name, train_id, checkpoint_name, 
            epochs, datetime.now().strftime(r'%m%d_%H%M%S'))
    model_script_subfolder = os.path.join(model_script_folder, train_id)
    model_script_path = os.path.join(model_script_folder, train_id, model_script_name)
    print('Created libtorch script', model_script_path)
    
    save_dir = Path(model_script_subfolder)
    save_dir.mkdir(parents=True, exist_ok=True)
    sm.save(model_script_path)


def create_all_libtorch_scripts(train_id, logs, experiment_name, model_name=None, model_script_folder='./libtorch-models'):
    model_paths = get_trained_model_paths(train_id, logs, experiment_name)
    config = load_model_config(experiment_name, train_id, logs)
    
    # In case weights are used for a different model. In this is only necessary when poor software
    #  engineering practices leads to refactoring names of models
    if model_name is not None:
        config["arch"]["type"] = model_name

    for path in model_paths:
        model, epochs = load_trained_model_by_path(path, config)

        s = re.findall("([^/]+)\.pth", path)
        assert s is not None, 'file in path {} does not exist'.format(path)
        checkpoint_name = s[0]
        _create_libtorch_script_from_model(model, epochs, train_id, experiment_name, checkpoint_name,
                model_script_folder)


# Load a libtorch script file
def load_libtorch_script(model_script_path):
    sm_loaded = torch.jit.load(model_script_path)
    sm_loaded.eval()
    print('Loaded libtorch script', model_script_path)
    return sm_loaded


def get_trained_model_paths(train_id, logs, experiment_name):
    files = os.path.join(logs, 'models', experiment_name, train_id, '*.pth')
    return get_sorted_files(files)


def get_sorted_files(files):
    """Given a Unix-style pathname pattern, returns a sorted list of files matching that pattern"""
    files = glob.glob(files)
    files.sort()
    return files

