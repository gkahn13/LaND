import tensorflow as tf

from sidewalk.utils import abstract
from sidewalk.models.cnn_model import CnnModel
from sidewalk.utils import tf_utils
from sidewalk.utils.python_utils import AttrDict


class RnnModel(CnnModel):
    __metaclass__ = abstract.BaseClass

    ############
    ### Init ###
    ############

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        self._horizon = params.horizon

    @abstract.overrides
    def _init_setup(self):
        super()._init_setup()
        self._init_rnn()

    def _init_rnn(self):
        assert self._obs_lowd_dim % 2 == 0
        self._rnn_dim = self._obs_lowd_dim // 2

        self._action_dim = self._env_spec.dim(self._env_spec.action_names)
        for name in self._observation_im_names + self._observation_vec_names:
            assert name in self._env_spec.observation_names

        if self._is_rnn_input_actions:
            self._rnn_input_model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', name='action/dense0'),
            tf.keras.layers.Dense(16, activation='relu', name='action/dense1'),
        ])

        self._rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(self._horizon, self._rnn_dim)

        self._rnn_output_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', name='output/dense0'),
            tf.keras.layers.Dense(self._rnn_output_dim,
                                  activation=None, name='output/dense1'),
        ])

    ##################
    ### Properties ###
    ##################

    @property
    @abstract.abstractmethod
    def _rnn_output_dim(self):
        raise NotImplementedError

    @property
    @abstract.abstractmethod
    def _is_rnn_input_actions(self):
        raise NotImplementedError

    ###########
    ### Run ###
    ###########

    def _get_rnn_outputs(self, obs_lowd, actions, training=False):
        if self._is_rnn_input_actions:
            actions_whitened = self._concat_and_whiten(actions, self._env_spec.action_names)
            rnn_inputs = self._rnn_input_model(actions_whitened, training=training)
        else:
            rnn_inputs = tf.tile(obs_lowd[:, tf.newaxis], (1, self._horizon, 1))

        initial_state_c, initial_state_h = tf.split(obs_lowd, 2, axis=1)
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_state_c[tf.newaxis], initial_state_h[tf.newaxis])
        rnn_inputs_time_major = tf.transpose(rnn_inputs, (1, 0, 2))
        rnn_outputs_time_major, _ = self._rnn_cell(rnn_inputs_time_major, initial_state=initial_state)
        rnn_outputs_batch_major = tf.transpose(rnn_outputs_time_major, (1, 0, 2))
        rnn_outputs = self._rnn_output_model(rnn_outputs_batch_major, training=training)

        return rnn_outputs

    @abstract.abstractmethod
    def _get_outputs(self, inputs, rnn_outputs):
        raise NotImplementedError

    @abstract.overrides
    def call(self, inputs, obs_lowd=None, training=False):
        obs_lowd = obs_lowd if obs_lowd is not None else self.get_obs_lowd(inputs, training=training)
        actions = inputs.leaf_filter(lambda k, v: k in self._env_spec.action_names)
        rnn_outputs = self._get_rnn_outputs(obs_lowd, actions, training=training)
        outputs = self._get_outputs(inputs, rnn_outputs, training=training)
        if training:
            outputs.kernels = tf_utils.get_kernels(self.layers)

        return outputs.freeze()
