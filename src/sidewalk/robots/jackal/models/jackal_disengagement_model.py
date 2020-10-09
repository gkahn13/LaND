import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sidewalk.models.rnn_model import RnnModel
from sidewalk.utils import abstract
from sidewalk.utils.python_utils import AttrDict


class JackalDisengagementModel(RnnModel):

    ############
    ### Init ###
    ############

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        self._finetune_cnn = params.finetune_cnn
        self._init_with_imagenet_weights = params.init_with_imagenet_weights
        self._num_collision_bins = params.num_collision_bins

    @abstract.overrides
    def _init_cnn(self):
        assert self._has_observation_im_inputs
        self._obs_im_model = tf.keras.Sequential([
            tf.keras.applications.MobileNetV2(input_shape=self._env_spec.names_to_shapes.images.front,
                                              weights='imagenet' if self._init_with_imagenet_weights else None,
                                              include_top=False), # 2M
            tf.keras.layers.Flatten(),
        ])
        self._obs_im_model.trainable = self._finetune_cnn

        if self._has_observation_vec_inputs:
            self._obs_vec_model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', name='obs_vec/dense0'),
                tf.keras.layers.Dense(32, activation=None, name='obs_vec/dense1'),
            ])
        else:
            self._obs_vec_model = None

        if self._has_observation_im_inputs or self._has_observation_vec_inputs:
            self._obs_lowd_model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', name='obs_lowd/dense0'),
                tf.keras.layers.Dense(128, activation='relu', name='obs_lowd/dense1'),
                tf.keras.layers.Dense(self._obs_lowd_dim, activation=None, name='obs_lowd/dense2'),
            ])
        else:
            self._obs_lowd_model = None

    ##################
    ### Properties ###
    ##################

    @property
    @abstract.overrides
    def _rnn_output_dim(self):
        return self._num_collision_bins

    @property
    @abstract.overrides
    def _is_rnn_input_actions(self):
        return True

    ###########
    ### Run ###
    ###########

    @abstract.overrides
    def _get_outputs(self, inputs, rnn_outputs, training=False):
        # bin the actions
        actions = inputs.commands.turn[..., 0]
        lower, upper = -1., 1.
        edges = np.linspace(lower, upper, self._num_collision_bins + 1).astype(np.float32)
        bins = tf.cast(tfp.stats.find_bins(actions, edges, extend_lower_interval=True, extend_upper_interval=True), tf.int32)

        turn_one_hot = tf.one_hot(bins, depth=self._num_collision_bins, axis=2)

        # for training
        all_pre_logits = rnn_outputs
        all_unscaled_logits = tf.nn.softmax(all_pre_logits, axis=-1)
        all_logits = -5 + 10 * all_unscaled_logits # from -5 to 5
        logits = tf.reduce_sum(all_logits * turn_one_hot, axis=-1)
        logits = logits[..., tf.newaxis] # for compatibility

        # for planning
        probs = tf.nn.sigmoid(logits)

        return AttrDict(
            turn_one_hot=turn_one_hot,
            logits=logits,
            probcoll=probs,

            # debuggin
            all_logits=all_logits
        )
