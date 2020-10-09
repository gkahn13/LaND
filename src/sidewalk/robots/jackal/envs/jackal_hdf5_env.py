import numpy as np
import os

from sidewalk.envs.data_env import DataEnv
from sidewalk.robots.jackal.data.hdf5_io import Hdf5Traverser
from sidewalk.robots.jackal.utils import constants
from sidewalk.robots.jackal.utils.jackal_utils import process_image
from sidewalk.utils import abstract, file_utils
from sidewalk.utils.python_utils import AttrDict


class JackalHdf5Env(DataEnv):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._hdf5_folders = params.hdf5_folders
        self._start_timestep = params.get('start_timestep', 0)
        self._shift_angle = params.get('shift_angle', 0.)

    @abstract.overrides
    def _init_setup(self):
        hdf5_fnames = file_utils.get_files_ending_with(self._hdf5_folders, '.hdf5')
        self._data_traverser = Hdf5Traverser(hdf5_fnames)
        for _ in range(self._start_timestep):
            self._data_traverser.next_timestep()

    @abstract.overrides
    def _get_obs(self):
        obs = AttrDict()

        for name in set(self._env_spec.observation_names):
            obs[name] = self._data_traverser.get(name, horizon=1)[0]

        original_image = obs.images.front
        desired_shape = self._env_spec.names_to_shapes.images.front
        image = process_image(original_image, desired_shape, image_rectify=True)
        obs.images.front = image

        return obs

    @abstract.overrides
    def _get_goal(self):
        goal = AttrDict()

        goal.is_turn = np.ravel(False).astype(np.float32)
        goal.turn_goal = np.ravel(0.).astype(np.float32)

        return goal

    @abstract.overrides
    def _get_done(self):
        return False