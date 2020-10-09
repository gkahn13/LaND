import numpy as np
import tensorflow as tf

from sidewalk.experiments import logger
from sidewalk.policies.eager_cem import EagerCEMPolicy
from sidewalk.utils import abstract
from sidewalk.utils.python_utils import AttrDict


class StaticCEMPolicy(EagerCEMPolicy):

    @abstract.overrides
    def _init_setup(self):
        super()._init_setup()

        # static graph
        self._session = None
        self._obs_placeholders = None
        self._goal_placeholders = None
        self._get_action_outputs = None

    @abstract.overrides
    def warm_start(self, model, observation, goal):
        assert not tf.executing_eagerly()

        logger.debug('Setting up CEM graph....')
        self._session = tf.get_default_session()
        assert self._session is not None

        ### create placeholders
        self._obs_placeholders = AttrDict()
        for name in self._env_spec.observation_names:
            shape = list(self._env_spec.names_to_shapes[name])
            dtype = tf.as_dtype(self._env_spec.names_to_dtypes[name])
            ph = tf.placeholder(dtype, shape=shape, name=name)
            self._obs_placeholders[name] = ph

        self._goal_placeholders = AttrDict()
        for name, value in goal.leaf_items():
            self._goal_placeholders[name] =  tf.placeholder(tf.as_dtype(value.dtype), shape=value.shape, name=name)

        self._get_action_outputs = self._cem(model, self._obs_placeholders, self._goal_placeholders)

        logger.debug('CEM graph setup complete')

    def _get_action_feed_dict(self, observation, goal):
        feed_dict = {}
        for name, ph in self._obs_placeholders.leaf_items():
            value = np.array(observation[name])
            if value.shape == tuple():
                value = value[np.newaxis]
            feed_dict[ph] = value
        for name, ph in self._goal_placeholders.leaf_items():
            feed_dict[ph] = np.array(goal[name])

        return feed_dict

    @abstract.overrides
    def get_action(self, model, observation, goal):
        assert self._session is not None

        feed_dict = self._get_action_feed_dict(observation, goal)

        get_action_tf = {}
        for name, tensor in self._get_action_outputs.leaf_items():
            get_action_tf[name] = tensor

        get_action_tf_output = self._session.run(get_action_tf, feed_dict=feed_dict)

        get_action = AttrDict.from_dict(get_action_tf_output)
        get_action.cost_fn = self._cost_fn

        return get_action
