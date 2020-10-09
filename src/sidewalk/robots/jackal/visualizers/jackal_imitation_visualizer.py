import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from sidewalk.robots.jackal.utils import jackal_utils
from sidewalk.robots.jackal.utils import constants
from sidewalk.utils import abstract, pyblit
from sidewalk.utils.python_utils import AttrDict
from sidewalk.visualizers.visualizer import Visualizer


class JackalImitationVisualizer(Visualizer):

    @abstract.overrides
    def _init_setup(self):
        self._fig, axes = plt.subplots(4, 1, figsize=(8, 20))
        ax_observation, ax_policy, ax_turn, ax_model = axes.ravel()

        self._pyblit = AttrDict(
            observation=self._init_setup_observation(ax_observation),
            policy=self._init_setup_policy(ax_policy),
            turn=self._init_setup_turn(ax_turn),
            model=self._init_setup_model(ax_model)
        )

        self._fig_shown = False

    def _init_setup_observation(self, ax):
        imshow = pyblit.Imshow(ax)
        return AttrDict(
            imshow=imshow,
            ax=pyblit.Axis(ax, [imshow])
        )

    def _init_setup_policy(self, ax):
        batch_line = pyblit.BatchLineCollection(ax)
        return AttrDict(
            batch_line=batch_line,
            ax=pyblit.Axis(ax, [batch_line])
        )

    def _init_setup_turn(self, ax):
        bar = pyblit.Barh(ax)
        return AttrDict(
            bar=bar,
            ax=pyblit.Axis(ax, [bar])
        )

    def _init_setup_model(self, ax):
        imshow = pyblit.Imshow(ax)
        return AttrDict(
            imshow=imshow,
            ax=pyblit.Axis(ax, [imshow])
        )

    @abstract.overrides
    def step(self, model, observation, goal, done, get_action):
        self._step_observation(observation)
        self._step_policy(get_action)
        self._step_turn(get_action)
        self._step_model(get_action)

        if not self._fig_shown:
            self._fig.tight_layout(pad=3.0)
            plt.show(block=False)
            plt.pause(0.1)
        self._fig_shown = True

    def _plot_positions(self, positions):
        forward_is_up = np.array([[0., -1.], [1., 0.]])
        return forward_is_up.dot(positions.T).T

    def _setup_batchline_axis(self, ax, horizon):
        max_step = constants.METERS_PER_STEP
        xlim = horizon * max_step * np.array([-1.0, 1.0])
        ylim = (-0.1 * max_step, horizon * max_step)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('meters')
        ax.set_ylabel('meters')

    def _step_observation(self, observation):
        image = observation.images.front
        self._pyblit.observation.imshow.draw(image)
        self._pyblit.observation.ax.draw()

    def _step_policy(self, get_action):
        turns = -get_action.action_sequence.commands.turn.ravel()
        steps = constants.METERS_PER_TIMESTEP * np.ones(np.shape(turns))
        positions = jackal_utils.turns_and_steps_to_positions(turns, steps)
        plot_positions = self._plot_positions(positions)

        xs = [plot_positions[:, 0]]
        ys = [plot_positions[:, 1]]
        colors = [(0, 0, 0, 1)] * (len(plot_positions) - 1)
        linewidths = [5.0] * (len(plot_positions) - 1)

        self._pyblit.policy.batch_line.draw(xs, ys, color=colors, linewidth=linewidths)

        self._pyblit.policy.ax.ax.set_title('Optimal action sequence')
        self._setup_batchline_axis(self._pyblit.policy.ax.ax, horizon=len(turns))
        self._pyblit.policy.ax.draw()

    def _step_turn(self, get_action):
        horizon = len(list(get_action.action_sequence.leaf_values())[0])

        ys = np.arange(horizon)
        widths = get_action.action_sequence.commands.turn.ravel()
        colors = ['k'] * horizon
        heights = [0.4] * horizon

        if 'action_sequence_var' in get_action.keys():
            # prepend so it's underneath the mean
            std = np.sqrt(get_action.action_sequence_var.commands.turn.ravel())
            ys = np.concatenate([ys + 0.02, ys])
            widths = np.concatenate([widths + np.sign(widths) * 0.5 * std, widths])
            colors = [(0., 0., 0., 0.5)] * horizon + colors
            heights = [0.15] * horizon + heights

        self._pyblit.turn.bar.draw(ys, widths, color=colors, height=heights)

        ax = self._pyblit.turn.ax.ax
        ax.set_xlim(2.0 * np.array(self._env_spec.names_to_limits.commands.turn))
        ax.set_xlabel('rad / time step')
        ax.set_ylabel('planning timestep')
        ax.set_yticks(np.arange(horizon))
        ax.set_title('Optimized actions')
        self._pyblit.turn.ax.draw()

    def _step_model(self, get_action):
        logits = get_action.all_model_outputs.logits[0]
        probs = softmax(logits)
        probs_rescaled = (probs - probs.min()) / (probs.max() - probs.min())
        self._pyblit.model.ax.ax.set_title('Action distribution')
        self._pyblit.model.imshow.draw(probs_rescaled[np.newaxis], cmap=plt.cm.magma, vmin=0., vmax=1.)
        self._pyblit.model.ax.draw()
