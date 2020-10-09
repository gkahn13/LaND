import numpy as np
import tensorflow as tf

from sidewalk.datasets.hdf5_dataset import Hdf5Dataset
from sidewalk.robots.jackal.utils.jackal_utils import process_image
from sidewalk.utils import abstract


class JackalHdf5Dataset(Hdf5Dataset):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        self._rebalance = params.get('rebalance', True)

        self._truncate_each_rollout_by = params.get('truncate_each_rollout_by', None)
    @abstract.overrides
    def _init_setup(self):
        super()._init_setup()

        ### collision / no collision indices

        collision = self._datadict.collision.ravel().copy()
        collision_horizon = np.convolve(collision, [1.] * self._horizon + [0.] * (self._horizon - 1), mode='same') > 0
        nocollision_horizon = np.logical_not(collision_horizon)

        done = self._datadict.done
        collision_horizon = np.logical_and(collision_horizon, np.logical_not(done))[:-self._horizon]
        nocollision_horizon = np.logical_and(nocollision_horizon, np.logical_not(done))[:-self._horizon]

        self._collision_indices = np.where(collision_horizon)[0]
        self._nocollision_indices = np.where(nocollision_horizon)[0]

    @abstract.overrides
    def _parse_hdf5(self, key, value):
        new_value = super()._parse_hdf5(key, value)
        if key == 'images/front':
            desired_shape = self._env_spec.names_to_shapes.images.front
            new_value = process_image(new_value, desired_shape, image_rectify=True)
        return new_value

    @abstract.overrides
    def _load_hdf5s(self):
        datadict, valid_start_indices = super()._load_hdf5s()

        if self._truncate_each_rollout_by is not None:
            done = datadict.done
            done_padded = np.convolve(
                done,
                [1] * self._truncate_each_rollout_by + [0] * (self._truncate_each_rollout_by - 1),
                mode='same'
            ) > 0
            keep = np.logical_not(
                np.convolve(
                    done,
                    [1] * (self._truncate_each_rollout_by - 1) + [0] * (self._truncate_each_rollout_by - 2),
                    mode='same'
                ) > 0
            )

            datadict.done = done_padded
            datadict.leaf_modify(lambda x: x[keep])
            valid_start_indices = np.where(np.logical_not(datadict.done))[0]

        return datadict, valid_start_indices

    @abstract.overrides
    def get_batch(self, indices=None, is_tf=True):
        if indices is None:
            if self._rebalance:
                assert len(self._collision_indices) > 0
                assert len(self._nocollision_indices) > 0

                collision_indices = np.random.choice(self._collision_indices, size=self._batch_size // 2)
                nocollision_indices = np.random.choice(self._nocollision_indices, size=self._batch_size - self._batch_size // 2)

                indices = np.concatenate([collision_indices, nocollision_indices], axis=0)
            else:
                indices = np.random.choice(self._valid_start_indices[:-self._horizon], size=self._batch_size)

        inputs, outputs = super().get_batch(indices=indices, is_tf=is_tf)

        ### replace actions after done with random samples from the training distribution
        assert is_tf
        batch_size = len(indices)
        done = tf.concat([tf.zeros([batch_size, 1], dtype=tf.bool), outputs.done[:, :-1]], axis=1)
        done_float = tf.cast(done, tf.float32)
        turn = inputs.commands.turn[..., 0]
        random_turn = np.random.choice(self._datadict.commands.turn[..., 0], size=int(np.prod(turn.shape))).reshape(turn.shape)
        turn = (1 - done_float) * turn + done_float * random_turn
        inputs.commands.turn = turn[..., tf.newaxis]

        return inputs, outputs
