import tensorflow as tf

from sidewalk.experiments import logger
from sidewalk.utils import abstract


class Model(tf.keras.Model):
    __metaclass__ = abstract.BaseClass

    def __init__(self, params, env_spec):
        super(Model, self).__init__()

        self._env_spec = env_spec
        self._init_params_to_attrs(params)
        self._init_setup()

    @abstract.abstractmethod
    def _init_params_to_attrs(self, params):
        pass

    @abstract.abstractmethod
    def _init_setup(self):
        pass

    @abstract.abstractmethod
    def get_obs_lowd(self, inputs, training=False):
        """
        :param inputs (AttrDict):
        :param training (bool):
        :return: Tensor or list
        """
        raise NotImplementedError

    @abstract.abstractmethod
    def call(self, inputs, obs_lowd=None, training=False):
        """
        :param inputs (AttrDict):
        :param obs_lowd
        :param training (bool):
        :return: AttrDict
        """
        raise NotImplementedError

    def restore(self, fname):
        logger.debug('Restoring model {0}'.format(fname))
        assert tf.train.checkpoint_exists(fname)
        checkpointer = tf.train.Checkpoint(model=self)
        status = checkpointer.restore(fname)
        if not tf.executing_eagerly():
            status.initialize_or_restore(tf.get_default_session())
