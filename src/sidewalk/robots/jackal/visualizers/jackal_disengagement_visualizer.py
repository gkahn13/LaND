import matplotlib.pyplot as plt
import numpy as np

from sidewalk.robots.jackal.utils import constants, jackal_utils
from sidewalk.robots.jackal.visualizers.jackal_imitation_visualizer import JackalImitationVisualizer
from sidewalk.utils import abstract, pyblit
from sidewalk.utils.python_utils import AttrDict


class JackalDisengagementVisualizer(JackalImitationVisualizer):

    @abstract.overrides
    def _init_setup_model(self, ax):
        batch_line = pyblit.BatchLineCollection(ax)
        return AttrDict(
            batch_line=batch_line,
            ax=pyblit.Axis(ax, [batch_line])
        )

    @abstract.overrides
    def _step_model(self, get_action):
        probcolls = get_action.all_model_outputs.probcoll[..., 0]
        idxs = np.argsort(get_action.all_costs)

        # subsample
        subsample = 10
        idxs = idxs[np.linspace(0, len(idxs) - 1, subsample).astype(int)]
        probcolls = probcolls[idxs]
        turns = -get_action.all_actions.commands.turn[idxs, ..., 0]

        # positions
        steps = constants.METERS_PER_TIMESTEP * np.ones(np.shape(turns))
        positions = jackal_utils.turns_and_steps_to_positions(turns, steps)
        plot_positions = np.array([self._plot_positions(p) for p in positions])

        # plot
        xs = plot_positions[..., 0]
        ys = plot_positions[..., 1]
        colors = plt.cm.autumn_r(probcolls)
        linewidths = 1.5 * np.ones(len(xs))

        self._pyblit.model.ax.ax.set_title('Planner')
        self._pyblit.model.batch_line.draw(xs, ys, color=colors, linewidth=linewidths)
        self._setup_batchline_axis(self._pyblit.model.ax.ax, horizon=turns.shape[1])
        self._pyblit.model.ax.draw()
