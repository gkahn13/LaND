import h5py
import numpy as np
import os
import tensorflow as tf

from sidewalk.datasets.dataset import Dataset
from sidewalk.experiments import logger
from sidewalk.utils import abstract, file_utils, np_utils
from sidewalk.utils.python_utils import AttrDict


class Hdf5Dataset(Dataset):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._hdf5_folders = params.hdf5_folders
        self._batch_size = params.batch_size
        self._horizon = params.horizon

        for folder in self._hdf5_folders:
            assert os.path.exists(folder)

    @abstract.overrides
    def _init_setup(self):
        self._datadict, self._valid_start_indices = self._load_hdf5s()

    def _parse_hdf5(self, key, value):
        new_value = np.array(value)
        if type(new_value[0]) == np.bytes_:
            new_value = np_utils.bytes2im(new_value)
        if new_value.dtype == np.float64:
            new_value = new_value.astype(np.float32)
        if len(new_value.shape) == 1:
            new_value = new_value[:, np.newaxis]

        dtype = self._env_spec.names_to_dtypes[key]
        return new_value.astype(dtype)

    def _load_hdf5s(self):
        hdf5_fnames = file_utils.get_files_ending_with(self._hdf5_folders, '.hdf5')

        # initialize to empty lists
        datadict = AttrDict()
        for key in self._env_spec.names:
            datadict[key] = []
        datadict.done = []
        datadict.hdf5_fname = []
        datadict.rollout_timestep = []

        # concatenate each hdf5
        for hdf5_fname in hdf5_fnames:
            logger.debug('Loading ' + hdf5_fname)
            with h5py.File(hdf5_fname, 'r') as f:
                hdf5_names = file_utils.get_hdf5_leaf_names(f)
                hdf5_lens = np.array([len(f[name]) for name in hdf5_names])
                if len(hdf5_names) == 0:
                    logger.warning('Empty hdf5, skipping!')
                    continue
                if not np.all(hdf5_lens == hdf5_lens[0]):
                    logger.warning('data lengths not all the same, skipping!')
                    continue
                if hdf5_lens[0] == 0:
                    logger.warning('data lengths are 0, skipping!')
                    continue

                for key in self._env_spec.names:
                    assert key in f, '"{0}" not in env space names'.format(key)
                    value = self._parse_hdf5(key, f[key])
                    datadict[key].append(value)
                datadict.done.append([False] * (len(value) - 1) + [True])
                datadict.hdf5_fname.append([hdf5_fname] * len(value))
                datadict.rollout_timestep.append(np.arange(len(value)))

        # turn every value into a single numpy array
        datadict.leaf_modify(lambda arr_list: np.concatenate(arr_list, axis=0))
        datadict_len = len(datadict.done)
        datadict.leaf_assert(lambda arr: len(arr) == datadict_len)
        logger.debug('Dataset length: {}'.format(datadict_len))

        # everywhere not done
        valid_start_indices = np.where(np.logical_not(datadict.done))[0]

        return datadict, valid_start_indices

    @abstract.overrides
    def get_batch(self, indices=None, is_tf=True):
        if indices is None:
            indices = np.random.choice(self._valid_start_indices[:-self._horizon],
                                       size=self._batch_size)

        sampled_datadict = self._datadict.leaf_apply(
            lambda arr: np.stack([arr[idx:idx+self._horizon+1] for idx in indices], axis=0))

        inputs = AttrDict()
        outputs = AttrDict()
        for key in self._env_spec.names:
            value = sampled_datadict[key]

            if key in self._env_spec.observation_names:
                inputs[key] = value[:, 0]
            elif key in self._env_spec.action_names:
                inputs[key] = value[:, :-1]

            if key in self._env_spec.output_observation_names:
                outputs[key] = value[:, 1:]

        outputs.done = sampled_datadict.done[:, 1:].cumsum(axis=1).astype(bool)

        if is_tf:
            for d in (inputs, outputs):
                d.leaf_modify(lambda x: tf.convert_to_tensor(x))

        return inputs, outputs

    def __len__(self):
        return len(self._datadict.done)


if __name__ == '__main__':
    from sidewalk.envs.env_spec import EnvSpec
    d = Hdf5Dataset(AttrDict(), EnvSpec(AttrDict()))