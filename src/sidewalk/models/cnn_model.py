import tensorflow as tf

from sidewalk.models.model import Model
from sidewalk.utils import abstract, tf_utils
from sidewalk.utils.python_utils import AttrDict


class CnnModel(Model):
    __metaclass__ = abstract.BaseClass

    ############
    ### Init ###
    ############

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        self._observation_im_names = sorted(tuple(params.observation_im_names))
        self._observation_vec_names = sorted(tuple(params.observation_vec_names))
        self._obs_lowd_dim = params.obs_lowd_dim

    @abstract.overrides
    def _init_setup(self):
        self._init_cnn()

    def _init_cnn(self):
        if self._has_observation_im_inputs:
            self._obs_im_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu',
                                       name='obs_im/conv0'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu',
                                       name='obs_im/conv1'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu',
                                       name='obs_im/conv2'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu', name='obs_im/dense0'),
                tf.keras.layers.Dense(128, activation=None, name='obs_im/dense1'),
            ])
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
                tf.keras.layers.Dense(128, activation='relu', name='obs_lowd/dense0'),
                tf.keras.layers.Dense(self._obs_lowd_dim, activation=None, name='obs_lowd/dense1'),
            ])
        else:
            self._obs_lowd_model = None

    ##################
    ### Properties ###
    ##################

    @property
    def _has_observation_im_inputs(self):
        return len(self._observation_im_names) > 0

    @property
    def _has_observation_vec_inputs(self):
        return len(self._observation_vec_names) > 0

    ###############
    ### Helpers ###
    ###############

    def _concat_and_whiten(self, inputs, names):
        inputs_concat = tf.concat([inputs[name] for name in names], axis=-1)
        lower, upper = self._env_spec.limits(names)
        mean = 0.5 * (lower + upper)
        var = 0.5 * (upper - lower)
        inputs_concat_whitened = (inputs_concat - mean) / var

        return inputs_concat_whitened

    def _unwhiten_and_split(self, tensor_whitened, names):
        lower, upper = self._env_spec.limits(names)
        mean = 0.5 * (lower + upper)
        var = 0.5 * (upper - lower)
        tensor = tensor_whitened * var + mean

        return AttrDict.from_dict({k: v for k, v in
                                   zip(names,
                                       tf.split(tensor, self._env_spec.dims(names), axis=2))})

    ###########
    ### Run ###
    ###########

    @abstract.overrides
    def get_obs_lowd(self, inputs, training=False):
        # setup inputs
        inputs.leaf_modify(lambda x: x[..., tf.newaxis] if len(x.shape) == 1 else x)
        inputs.leaf_modify(lambda x: tf.cast(x, tf.float32))
        for name in self._env_spec.action_names:
            value = inputs[name]
            if len(value.shape) == 2:
                inputs[name] = value[..., tf.newaxis]

        ### observations
        obs_lowd_list = []

        if self._has_observation_im_inputs:
            obs_ims = tf.concat([inputs[name] for name in self._observation_im_names], axis=-1)
            obs_ims = (obs_ims / 255.) - 0.5
            obs_im_lowd = self._obs_im_model(obs_ims, training=training)
            obs_lowd_list.append(obs_im_lowd)

        if self._has_observation_vec_inputs:
            obs_vecs_whitened = self._concat_and_whiten(inputs, self._observation_vec_names)
            obs_vec_lowd = self._obs_vec_model(obs_vecs_whitened, training=training)
            obs_lowd_list.append(obs_vec_lowd)

        if len(obs_lowd_list) > 0:
            obs_lowd = self._obs_lowd_model(tf.concat(obs_lowd_list, axis=1), training=training)
        else:
            batch_size = list(inputs.leaf_values())[0].shape.as_list()[0]
            obs_lowd = tf.zeros([batch_size, self._obs_lowd_dim])

        return obs_lowd

    @abstract.overrides
    def call(self, inputs, obs_lowd=None, training=False):
        obs_lowd = obs_lowd if obs_lowd is not None else self.get_obs_lowd(inputs, training=training)
        outputs = AttrDict(obs_lowd=obs_lowd)
        if training:
            outputs.kernels = tf_utils.get_kernels(self.layers)

        return outputs.freeze()