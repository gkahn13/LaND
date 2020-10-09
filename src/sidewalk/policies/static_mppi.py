import numpy as np
import tensorflow as tf

from sidewalk.experiments import logger
from sidewalk.policies.static_cem import StaticCEMPolicy
from sidewalk.utils import abstract
from sidewalk.utils.python_utils import AttrDict


class StaticMPPIPolicy(StaticCEMPolicy):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        ### planning params
        self._horizon = params.horizon
        self._action_selection_limits = params.action_selection_limits
        self._cost_fn = params.cost_fn

        ### MPPI params
        self._sigma = params.sigma
        self._N = params.N
        self._gamma = params.gamma
        self._beta = params.beta

    @abstract.overrides
    def _init_setup(self):
        super()._init_setup()

        # static graph
        self._mppi_mean_placeholder = None
        self._mppi_mean_np = None

    def _setup_mppi_graph(self, model, goals):
        ### create placeholders
        obs_placeholders = AttrDict()
        for name in self._env_spec.observation_names:
            shape = list(self._env_spec.names_to_shapes[name])
            dtype = tf.as_dtype(self._env_spec.names_to_dtypes[name])
            ph = tf.placeholder(dtype, shape=shape, name=name)
            obs_placeholders[name] = ph

        goal_placeholders = AttrDict()
        for name, value in goals.leaf_items():
            goal_placeholders[name] =  tf.placeholder(tf.as_dtype(value.dtype), shape=value.shape, name=name)

        mppi_mean_placeholder = tf.placeholder(tf.float32, name='mppi_mean', shape=[self._horizon, self._action_dim])

        ### get obs lowd
        inputs = obs_placeholders.leaf_apply(lambda value: value[tf.newaxis])
        obs_lowd = model.get_obs_lowd(inputs)

        past_mean = mppi_mean_placeholder[0]
        shifted_mean = tf.concat([mppi_mean_placeholder[1:], mppi_mean_placeholder[-1:]], axis=0)

        # sample through time
        delta_limits = 0.5 * (self._action_selection_upper_limits - self._action_selection_lower_limits)
        eps = tf.random_normal(mean=0, stddev=self._sigma * delta_limits,
                               shape=(self._N, self._horizon, self._action_dim))
        actions = []
        for h in range(self._horizon):
            if h == 0:
                # action_h = self._beta * (shifted_mean[h, :] + eps[:, h, :]) + (1. - self._beta) * past_mean
                action_h = self._beta * (past_mean + eps[:, h, :]) + (1. - self._beta) * past_mean
            else:
                action_h = self._beta * (shifted_mean[h, :] + eps[:, h, :]) + (1. - self._beta) * actions[-1]
            actions.append(action_h)
        actions = tf.stack(actions, axis=1)
        actions = tf.clip_by_value(
            actions,
            self._action_selection_lower_limits[np.newaxis, np.newaxis],
            self._action_selection_upper_limits[np.newaxis, np.newaxis]
        )

        # forward simulate
        actions_split = self._split_action(actions)
        inputs_tiled = inputs.leaf_filter(lambda k, v: k in self._env_spec.output_observation_names).leaf_apply(
            lambda v: tf.tile(v, [self._N] + [1] * (len(v.shape) - 1))
        )
        inputs_tiled.combine(actions_split)
        inputs_tiled.freeze()
        obs_lowd_tiled = tf.tile(obs_lowd, (self._N, 1))

        ### call model and evaluate cost
        model_outputs = model.call(inputs_tiled, obs_lowd=obs_lowd_tiled)
        costs_per_timestep = self._cost_fn(inputs_tiled, model_outputs, goal_placeholders, actions_split)
        costs = tf.reduce_mean(costs_per_timestep.total, axis=1)

        # MPPI update
        scores = -costs
        probs = tf.exp(self._gamma * (scores - tf.reduce_max(scores)))
        probs /= tf.reduce_sum(probs) + 1e-10
        new_mppi_mean = tf.reduce_sum(actions * probs[:, tf.newaxis, tf.newaxis], axis=0)

        best_idx = tf.argmin(costs)
        best_actions = self._split_action(new_mppi_mean)

        get_action_outputs = AttrDict(
            cost=costs[best_idx],
            cost_per_timestep=costs_per_timestep.leaf_apply(lambda v: v[best_idx]),
            action=best_actions.leaf_apply(lambda v: v[0]),
            action_sequence=best_actions,
            # model_outputs=model_outputs.leaf_filter(lambda k, v: tf.is_tensor(v)).leaf_apply(lambda v: v[best_idx]),

            all_costs=costs,
            all_costs_per_timestep=costs_per_timestep,
            all_actions=actions_split,
            all_model_outputs=model_outputs.leaf_filter(lambda k, v: tf.is_tensor(v)),

            mppi_mean=new_mppi_mean,
        ).freeze()

        return obs_placeholders, goal_placeholders, mppi_mean_placeholder, get_action_outputs

    @abstract.overrides
    def warm_start(self, model, observation, goal):
        assert not tf.executing_eagerly()

        logger.debug('Setting up MPPI graph....')
        self._session = tf.get_default_session()
        assert self._session is not None

        self._obs_placeholders, self._goal_placeholders, self._mppi_mean_placeholder, self._get_action_outputs = \
            self._setup_mppi_graph(model, goal)
        self._mppi_mean_np = np.zeros([self._horizon, self._action_dim], dtype=np.float32)

        logger.debug('MPPI graph setup complete')

    @abstract.overrides
    def _get_action_feed_dict(self, observation, goal):
        feed_dict = super()._get_action_feed_dict(observation, goal)
        feed_dict[self._mppi_mean_placeholder] = self._mppi_mean_np
        return feed_dict

    @abstract.overrides
    def get_action(self, model, observation, goal):
        get_action = super().get_action(model, observation, goal)
        self._mppi_mean_np = get_action.mppi_mean

        return get_action
