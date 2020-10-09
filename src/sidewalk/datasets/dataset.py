from sidewalk.envs.env_spec import EnvSpec
from sidewalk.utils import abstract
from sidewalk.utils.python_utils import AttrDict


class Dataset(abstract.BaseClass):

    def __init__(self, params, env_spec):
        assert isinstance(params, AttrDict)
        assert isinstance(env_spec, EnvSpec)

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
    def get_batch(self):
        """
        Returns:
            inputs (AttrDict)
            outputs (AttrDict)
        """
        raise NotImplementedError
        inputs = AttrDict()
        outputs = AttrDict()
        return inputs, outputs
