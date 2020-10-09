import h5py
import numpy as np
import os
import threading
import time

from sidewalk.experiments import logger
from sidewalk.utils import abstract, data_utils, np_utils
from sidewalk.utils.python_utils import AttrDict

class Hdf5Base(abstract.BaseClass):

    def __init__(self, keys=None):
        self.keys = keys or (
            'images/front',
            'collision',
            'commands/turn',
            'commands/dt',
        )


class Hdf5Reader(Hdf5Base):

    def __init__(self, fname, keys=None):
        super().__init__(keys=keys)
        assert os.path.exists(fname)
        self._load_hdf5(fname)

    def _load_hdf5(self, fname):
        self._d = AttrDict()
        with h5py.File(fname, 'r') as f:
            for key in self.keys:
                self._d[key] = np.array(f[key])
                if 'image' in key:
                    self._d[key] = np.array(list(np_utils.uncompress_video(self._d[key])))

    def get(self, key, start=None, stop=None):
        return self._d[key][start:stop]

    def __len__(self):
        return len(self._d['commands/turn'])


class AsyncHdf5Reader(Hdf5Reader):

    @abstract.overrides
    def _load_hdf5(self, fname):
        self._d = AttrDict()
        self._image_buffer = None
        self._image_key = None
        with h5py.File(fname, 'r') as f:
            for key in self.keys:
                if 'image' in key:
                    assert self._image_key is None
                    self._image_key = key
                    self._image_buffer = np.array(f[key])
                    self._d[key] = list()
                else:
                    self._d[key] = np.array(f[key])

        assert self._image_buffer is not None
        thread = threading.Thread(target=self._background_video_thread)
        thread.daemon = True
        thread.start()

    def _background_video_thread(self):
        for frame in np_utils.uncompress_video(self._image_buffer):
            self._d[self._image_key].append(frame)

    @abstract.overrides
    def get(self, key, start=None, stop=None):
        start = start or 0
        stop = stop or len(self)

        if key == self._image_key:
            while start >= len(self._d[self._image_key]) or stop > len(self._d[self._image_key]):
                logger.info('Waiting for images to come in...')
                time.sleep(1)

        return np.array(super().get(key, start=start, stop=stop))


class Hdf5Traverser(data_utils.DataTraverser):

    def __init__(self, data_fnames, keys=None):
        self._keys = keys
        super().__init__(data_fnames)

    @abstract.overrides
    def get(self, key, horizon=None):
        start = self._curr_data_timestep
        end = min(start + horizon, self._curr_data_len)
        return self._curr_data.get(key, start, end)

    @abstract.overrides
    def _load_data(self):
        self._curr_data = AsyncHdf5Reader(self._data_fnames[self._curr_data_idx], keys=self._keys)
        self._curr_data_len = len(self._curr_data)
