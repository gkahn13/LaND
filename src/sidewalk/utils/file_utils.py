import glob
import h5py
import importlib.util
import itertools
import os
from pathlib import Path


def get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        folder = folder_or_folders
        assert os.path.exists(folder)

        fnames = []
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                fnames.append(os.path.join(folder, fname))
        return sorted(fnames)
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return sorted(list(itertools.chain(*[get_files_ending_with(folder, ext) for folder in folder_or_folders])))


def recursive_get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        return sorted([str(path) for path in Path(folder_or_folders).rglob('*{0}'.format(ext))])
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return sorted(list(itertools.chain(*[recursive_get_files_ending_with(folder, ext)
                                             for folder in folder_or_folders])))


def get_hdf5_leaf_names(node, name=''):
    if isinstance(node, str):
        assert os.path.exists(node)
        with h5py.File(node, 'r') as f:
            return get_hdf5_leaf_names(f)
    elif isinstance(node, h5py.Dataset):
        return [name]
    else:
        names = []
        for child_name, child in node.items():
            names += get_hdf5_leaf_names(child, name=name+'/'+child_name)
        return names


def import_config(config_fname):
    assert config_fname.endswith('.py')
    spec = importlib.util.spec_from_file_location('config', config_fname)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.params
