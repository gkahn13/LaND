import numpy as np
import tensorflow as tf

from sidewalk.policies.policy import Policy
from sidewalk.utils import abstract
from sidewalk.utils.python_utils import AttrDict


class EagerCEMPolicy(Policy):

    @abstract.overrides
    def _init_params_to_attrs(self, params):
        super()._init_params_to_attrs(params)

        ### planning params
        self._horizon = params.horizon
        self._action_selection_limits = params.action_selection_limits
        self._cost_fn = params.cost_fn

        # CEM specific params
        self._M_init = params.M_init
        self._M = params.M
        self._K = params.K
        self._itrs = params.itrs
        self._eps = params.eps

    @abstract.overrides
    def _init_setup(self):
        super()._init_setup()

        ### action limits
        lower, upper = [], []
        for name in self._env_spec.action_names:
            l, u = self._action_selection_limits[name]
            lower.append(np.ravel(l))
            upper.append(np.ravel(u))
        self._action_selection_lower_limits = np.ravel(list(zip(*lower))).astype(np.float32)
        self._action_selection_upper_limits = np.ravel(list(zip(*upper))).astype(np.float32)
        self._action_dim = self._env_spec.dim(self._env_spec.action_names)

    def _split_action(self, actions):
        split_actions = AttrDict()
        for name, tensor in zip(self._env_spec.action_names,
                                tf.split(actions, self._env_spec.dims(self._env_spec.action_names), axis=-1)):
            split_actions[name] = tensor
        return split_actions

    def _cem(self, model, observation, goal):
        observation.leaf_modify(lambda v: tf.convert_to_tensor(v))
        goal.leaf_modify(lambda v: tf.convert_to_tensor(v))

        ### get obs lowd
        inputs = observation.leaf_apply(lambda value: value[tf.newaxis])
        obs_lowd = model.get_obs_lowd(inputs)

        ### CEM setup
        inputs = inputs.leaf_filter(lambda key, value: len(value.shape) < 4)
        action_selection_lower_limits = np.tile(self._action_selection_lower_limits, (self._horizon,))
        action_selection_upper_limits = np.tile(self._action_selection_upper_limits, (self._horizon,))
        action_distribution = tf.contrib.distributions.Uniform(
            action_selection_lower_limits,
            action_selection_upper_limits
        )
        # CEM params
        Ms = [self._M_init] + [self._M] * (self._itrs - 1)
        Ks = [self._K] * (self._itrs - 1) + [1]

        ### keep track of
        all_costs = []
        all_costs_per_timestep = []
        all_actions = []
        all_model_outputs = []

        ### CEM loop
        for M, K in zip(Ms, Ks):
            concat_actions = tf.reshape(
                action_distribution.sample((M,)),
                (M, self._horizon, -1)
            )

            concat_actions = tf.clip_by_value(
                concat_actions,
                np.reshape(action_selection_lower_limits, (self._horizon, self._action_dim)),
                np.reshape(action_selection_upper_limits, (self._horizon, self._action_dim))
            )

            actions = self._split_action(concat_actions)

            inputs_tiled = inputs.leaf_filter(lambda k, v: k in self._env_spec.output_observation_names)
            inputs_tiled = inputs_tiled.leaf_apply(lambda v: tf.tile(v, [M] + [1] * (len(v.shape) - 1)))

            for k, v in actions.leaf_items():
                inputs_tiled[k] = v

            obs_lowd_tiled = tf.tile(obs_lowd, (M, 1))

            ### call model and evaluate cost
            model_outputs = model.call(inputs_tiled, obs_lowd=obs_lowd_tiled)
            model_outputs = model_outputs.leaf_filter(lambda key, value: key[0] != '_')
            costs_per_timestep = self._cost_fn(inputs_tiled, model_outputs, goal, actions)
            costs = tf.reduce_mean(costs_per_timestep.total, axis=1)

            ### keep track
            all_costs.append(costs)
            all_costs_per_timestep.append(costs_per_timestep)
            all_actions.append(actions)
            all_model_outputs.append(model_outputs.leaf_filter(lambda k, v: tf.is_tensor(v)))

            ### get top K
            _, top_indices = tf.nn.top_k(-costs, k=K)
            top_actions = tf.gather(
                tf.reshape(concat_actions, [M, self._horizon * self._action_dim]),
                indices=top_indices
            )

            ### set new distribution based on top k
            mean = tf.reduce_mean(top_actions, axis=0)
            covar = tf.matmul(tf.transpose(top_actions), top_actions) / float(K)
            sigma = covar + self._eps * tf.eye(self._horizon * self._action_dim)

            action_distribution = tf.contrib.distributions.MultivariateNormalFullCovariance(
                loc=mean,
                covariance_matrix=sigma
            )

        all_costs = tf.concat(all_costs, axis=0)
        all_costs_per_timestep = AttrDict.leaf_combine_and_apply(all_costs_per_timestep, lambda arrs: tf.concat(arrs, axis=0))
        all_actions = AttrDict.leaf_combine_and_apply(all_actions, lambda arrs: tf.concat(arrs, axis=0))
        all_model_outputs = AttrDict.leaf_combine_and_apply(all_model_outputs, lambda arrs: tf.concat(arrs, axis=0))

        best_idx = tf.argmin(all_costs)
        best_cost = all_costs[best_idx]
        best_cost_per_timestep = all_costs_per_timestep.leaf_apply(lambda v: v[best_idx])
        best_action_sequence = all_actions.leaf_apply(lambda v: v[best_idx])
        best_action = best_action_sequence.leaf_apply(lambda v: v[0])
        # best_model_outputs = all_model_outputs.leaf_apply(lambda v: v[best_idx])

        get_action_outputs = AttrDict(
            cost=best_cost,
            cost_per_timestep=best_cost_per_timestep,
            action=best_action,
            action_sequence=best_action_sequence,
            # model_outputs=best_model_outputs,

            all_costs=all_costs,
            all_costs_per_timestep=all_costs_per_timestep,
            all_actions=all_actions,
            all_model_outputs=all_model_outputs
        )

        return get_action_outputs

    @abstract.overrides
    def warm_start(self, model, observation, goal):
        assert tf.executing_eagerly()

    @abstract.overrides
    def get_action(self, model, observation, goal):
        get_action = self._cem(model, observation, goal)
        get_action.leaf_modify(lambda v: np.array(v))
        get_action.cost_fn = self._cost_fn

        return get_action
