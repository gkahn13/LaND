import matplotlib.pyplot as plt
import numpy as np

from sidewalk.robots.jackal.data.hdf5_io import Hdf5Traverser
from sidewalk.robots.jackal.utils import constants, jackal_utils
from sidewalk.utils import abstract, pyblit
from sidewalk.utils.python_utils import Getch


class JackalHDF5Visualizer(Hdf5Traverser):

    def __init__(self, hdf5_fnames, horizon):
        super().__init__(hdf5_fnames)

        self._horizon = horizon
        self._setup_visualization()

    def _setup_visualization(self):
        self._f, axes = plt.subplots(1, 3, figsize=(20, 5))
        ax_im, ax_poses, ax_turn = axes
        self._plot_is_showing = False

        self._pyblit_im = pyblit.Imshow(ax_im)
        self._pyblit_im_ax = pyblit.Axis(ax_im, [self._pyblit_im])

        self._pyblit_poses = pyblit.Line(ax_poses)
        self._pyblit_poses_ax = pyblit.Axis(ax_poses, [self._pyblit_poses])

        self._pyblit_turn_bar = pyblit.Barh(ax_turn)
        self._pyblit_turn_ax = pyblit.Axis(ax_turn, [self._pyblit_turn_bar])

    #####################
    ### Hdf5Traverser ###
    #####################

    @abstract.overrides
    def get(self, key, horizon=None):
        return super().get(key, horizon=horizon or self._horizon)

    #################
    ### Visualize ###
    #################

    def _plot_im(self, pyblit_im, pyblit_im_ax):
        images = self.get('images/front', horizon=1)
        images = images[::-1].reshape([-1] + list(images.shape[2:]))
        pyblit_im.draw(images)
        pyblit_im_ax.draw()

    def _plot_positions(self):
        turns = self.get('commands/turn')
        steps = constants.METERS_PER_TIMESTEP * np.ones(np.shape(turns))

        positions = jackal_utils.turns_and_steps_to_positions(turns, steps)
        self._pyblit_poses.draw(positions[:, 1], positions[:, 0])

        ax = self._pyblit_poses_ax.ax
        max_position = 1.05 * constants.METERS_PER_STEP * self._horizon
        ax.set_xlim((-max_position, max_position))
        ax.set_ylim((-0.1, max_position))
        ax.set_aspect('equal')

        self._pyblit_poses_ax.draw()

    def _plot_turn(self):
        turns = self.get('commands/turn')
        collisions = self.get('collision')

        turns = np.pad(turns, [0, max(self._horizon - len(turns), 0)])
        collisions = np.pad(collisions, [0, max(self._horizon - len(collisions), 0)], constant_values=collisions[-1])

        color = ['r' if c else '#1f77b4' for c in collisions]

        self._pyblit_turn_bar.draw(np.arange(len(turns)), turns, color=color)
        ax = self._pyblit_turn_ax.ax
        ax.set_xlim((-0.5, 0.5))
        ax.set_xlabel('turn')
        ax.set_ylabel('timestep')
        ax.set_yticks(np.arange(len(turns)))
        self._pyblit_turn_ax.draw()

    def _plot_step(self):
        steps = self.get('commands/step')
        self._pyblit_step_bar.draw(np.arange(len(steps)), steps)
        ax = self._pyblit_step_ax.ax
        ax.set_ylim((0., 1.5))
        ax.set_ylabel('step')
        ax.set_xlabel('timestep')
        ax.set_xticks(np.arange(len(steps)))
        self._pyblit_step_ax.draw()

    def _update_visualization(self):
        self._plot_im(self._pyblit_im, self._pyblit_im_ax)
        self._plot_positions()
        self._plot_turn()

        if not self._plot_is_showing:
            plt.show(block=False)
            self._plot_is_showing = True
            plt.pause(0.01)
        self._f.canvas.flush_events()

    ###########
    ### Run ###
    ###########

    def run(self):
        self._update_visualization()

        while True:
            print(str(self))
            self._update_visualization()

            char = Getch.getch()
            self._prev_data_idx = self._curr_data_idx

            if char == 'q':
                break
            elif char == 'e':
                self.next_timestep()
            elif char == 'w':
                self.prev_timestep()
            elif char == 'd':
                self.next_data()
            elif char == 's':
                self.prev_data()
            elif char >= '1' and char <= '9':
                for _ in range(int(char)):
                    self.next_timestep()
            elif char == 'c':
                self.next_data_end()
            elif char == 'x':
                self.prev_data_end()
            else:
                continue
