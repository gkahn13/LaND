import sys

from sidewalk.envs.env import Env
from sidewalk.utils import abstract
from sidewalk.utils.python_utils import AttrDict, Getch


class DataEnv(Env):

    @abstract.overrides
    def _init_setup(self):
        self._data_traverser = None
        # self._data_traverser = data_utils.DataTraverser(data_fnames)

    def _get_obs(self):
        obs = AttrDict()
        for name in self._env_spec.observation_names:
            obs[name] = self._data_traverser.get(name, horizon=1)
        return obs

    def _get_goal(self):
        goal = AttrDict()
        for name in self._env_spec.goal_names:
            goal[name] = self._data_traverser.get(name, horizon=1)
        return goal

    @abstract.abstractmethod
    def _get_done(self):
        pass

    @abstract.overrides
    def step(self, action):
        print(str(self._data_traverser))
        char = Getch.getch()

        if char == 'q':
            sys.exit(0)
        elif char == 'e':
            self._data_traverser.next_timestep()
        elif char == 'w':
            self._data_traverser.prev_timestep()
        elif char == 'd':
            self._data_traverser.next_data()
        elif char == 's':
            self._data_traverser.prev_data()
        elif char >= '1' and char <= '9':
            for _ in range(int(char)):
                self._data_traverser.next_timestep()
        elif char == 'c':
            self._data_traverser.next_data_end()
        elif char == 'x':
            self._data_traverser.prev_data_end()
        else:
            pass

        obs = self._get_obs()
        goal = self._get_goal()
        done = self._get_done()
        return obs, goal, done

    @abstract.overrides
    def reset(self):
        obs = self._get_obs()
        goal = self._get_goal()
        self._data_traverser.next_timestep()
        return obs, goal
