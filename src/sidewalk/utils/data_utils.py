import os

from sidewalk.utils import abstract


class DataTraverser(abstract.BaseClass):

    def __init__(self, data_fnames):
        for fname in data_fnames:
            assert os.path.exists(fname)

        self._data_fnames = data_fnames

        self._curr_data_idx = 0
        self._prev_data_idx = -1
        self._curr_data_timestep = 0
        self._curr_data = None
        self._curr_data_len = None
        self._load_data()

    @property
    def curr_data_len(self):
        return self._curr_data_len

    @abstract.abstractmethod
    def get(self, key, start=None, horizon=None):
        raise NotImplementedError

    @abstract.abstractmethod
    def _load_data(self):
        raise NotImplementedError
        self._curr_data = None
        self._curr_data_len = None

    @property
    def no_more_timesteps(self):
        return self._curr_data_timestep >= self._curr_data_len - 1

    @property
    def no_more_files(self):
        return self._curr_data_idx == len(self._data_fnames) - 1

    def next_timestep(self):
        if self.no_more_timesteps and self.no_more_files:
            return # at the end, do nothing

        if self.no_more_timesteps:
            self._curr_data_idx += 1
            self._load_data()
            self._curr_data_timestep = 0
        else:
            self._curr_data_timestep += 1

    def prev_timestep(self):
        if (self._curr_data_timestep == 0) and (self._curr_data_idx == 0):
            return # at the beginning, do nothing

        self._curr_data_timestep -= 1
        if self._curr_data_timestep < 0:
            self._curr_data_idx -= 1
            self._load_data()
            self._curr_data_timestep = self._curr_data_len - 1

    def next_data(self):
        if self.no_more_files:
            return # at the end, do nothing

        self._curr_data_idx += 1
        self._load_data()
        self._curr_data_timestep = 0

    def prev_data(self):
        if self._curr_data_idx == 0:
            return # at the beginning, do nothing

        self._curr_data_idx -= 1
        self._load_data()
        self._curr_data_timestep = 0

    def next_data_end(self):
        if self.no_more_files:
            pass # at the end, do nothing
        else:
            self._curr_data_idx += 1

        self._load_data()
        self._curr_data_timestep = self._curr_data_len - 1

    def prev_data_end(self):
        if self._curr_data_idx == 0:
            pass  # at the end, do nothing
        else:
            self._curr_data_idx -= 1

        self._load_data()
        self._curr_data_timestep = self._curr_data_len - 1

    def __str__(self):
        return '{0}/{1}, {2}/{3} -- {4}'.format(self._curr_data_timestep + 1, self._curr_data_len,
                                                self._curr_data_idx + 1, len(self._data_fnames),
                                                os.path.basename(self._data_fnames[self._curr_data_idx]))
