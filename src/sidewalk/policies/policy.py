from sidewalk.utils import abstract


class Policy(abstract.BaseClass):

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
    def warm_start(self, model, observation, goal):
        raise NotImplementedError

    @abstract.abstractmethod
    def get_action(self, model, observation, goal):
        """
        Args:
            model (Model):
            observation (AttrDict):
            goal (AttrDict):

        Returns:
            AttrDict
        """
        raise NotImplementedError
