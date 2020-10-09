from sidewalk.utils import abstract


class Env(abstract.BaseClass):

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        self._init_params_to_attrs(params)
        self._init_setup()

    @abstract.abstractmethod
    def _init_params_to_attrs(self, params):
        pass

    @abstract.abstractmethod
    def _init_setup(self):
        pass

    @abstract.abstractmethod
    def step(self, action):
        raise NotImplementedError
        return obs, goal, done

    @abstract.abstractmethod
    def reset(self):
        raise NotImplementedError
        return obs, goal
