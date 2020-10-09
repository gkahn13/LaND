import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sidewalk.models.cnn_model import CnnModel
from sidewalk.utils import abstract, tf_utils
from sidewalk.utils.python_utils import AttrDict


class JackalImitationModel(CnnModel):

    ############
    ### Init ###
    ############

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._finetune_cnn = params.finetune_cnn
        self._init_with_imagenet_weights = params.init_with_imagenet_weights

        assert self._env_spec.action_names == ('commands/turn',)
        self._bin_edges = np.array(params.bin_edges).astype(np.float32)

        self._observation_im_names = sorted(tuple(params.observation_im_names))
        self._observation_vec_names = sorted(tuple(params.observation_vec_names))
        self._obs_lowd_dim = len(self._bin_edges)

    @abstract.overrides
    def _init_cnn(self):
        if self._has_observation_im_inputs:

            self._obs_im_model = tf.keras.Sequential([
                tf.keras.applications.MobileNetV2(input_shape=self._env_spec.names_to_shapes.images.front,
                                                  weights='imagenet' if self._init_with_imagenet_weights else None,
                                                  include_top=False),  # 2M
                tf.keras.layers.Flatten(),
            ])
            self._obs_im_model.trainable = self._finetune_cnn
        else:
            self._obs_im_model = None

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

    ###########
    ### Run ###
    ###########

    def _get_outputs(self, inputs, obs_lowd, training=False):
        # labels
        horizon = 1
        turn = tf.reshape(inputs.commands.turn[..., :horizon, 0], (-1,))
        bins = tfp.stats.find_bins(turn, self._bin_edges, extend_lower_interval=True, extend_upper_interval=True)

        # prediction
        logits = obs_lowd
        dist = tf.distributions.Categorical(logits)
        log_probs = dist.log_prob(bins)

        # accuracy
        accs = tf.cast(tf.equal(tf.argmax(logits, axis=-1), tf.cast(bins, tf.int64)), tf.float32)

        return AttrDict(
            logits=logits,
            log_prob=log_probs,
            acc=accs,
            bin_edges=tf.convert_to_tensor(self._bin_edges),
        )

    @abstract.overrides
    def call(self, inputs, obs_lowd=None, training=False):
        obs_lowd = obs_lowd if obs_lowd is not None else self.get_obs_lowd(inputs, training=training)
        outputs = self._get_outputs(inputs, obs_lowd, training=training)
        if training:
            outputs.kernels = tf_utils.get_kernels(self.layers)

        return outputs.freeze()