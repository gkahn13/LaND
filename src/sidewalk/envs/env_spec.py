import numpy as np

from sidewalk.utils import abstract
from sidewalk.utils.python_utils import AttrDict


class EnvSpec(abstract.BaseClass):

    def __init__(self, params, names_shapes_limits_dtypes=[]):
        names_shapes_limits_dtypes = list(names_shapes_limits_dtypes)
        names_shapes_limits_dtypes += [('done', (1,), (0, 1), np.bool)]

        self._names_to_shapes = AttrDict()
        self._names_to_limits = AttrDict()
        self._names_to_dtypes = AttrDict()
        for name, shape, limit, dtype in names_shapes_limits_dtypes:
            self._names_to_shapes[name] = shape
            self._names_to_limits[name] = limit
            self._names_to_dtypes[name] = dtype

    @property
    @abstract.abstractmethod
    def observation_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def output_observation_names(self):
        return self.observation_names

    @property
    def goal_names(self):
        """
        The only difference between a goal and an observation is that goals are user-specified

        Returns:
            list(str)
        """
        return tuple()

    @property
    @abstract.abstractmethod
    def action_names(self):
        """
        Returns:
            list(str)
        """
        raise NotImplementedError

    @property
    def names(self):
        """
        Returns:
            list(str)
        """
        return self.observation_names + self.goal_names + self.action_names

    @property
    def names_to_shapes(self):
        """
        Knowing the dimensions is useful for building neural networks

        Returns:
            AttrDict
        """
        return self._names_to_shapes

    @property
    def names_to_limits(self):
        """
        Knowing the limits is useful for normalizing data

        Returns:
            AttrDict
        """
        return self._names_to_limits

    @property
    def names_to_dtypes(self):
        """
        Knowing the data type is useful for building neural networks and datasets

        Returns:
            AttrDict
        """
        return self._names_to_dtypes

    def limits(self, names):
        lower, upper = [], []
        for name in names:
            shape = self.names_to_shapes[name]
            assert len(shape) == 1
            l, u = self.names_to_limits[name]
            lower += [l] * shape[0]
            upper += [u] * shape[0]
        return np.array(lower), np.array(upper)

    def dims(self, names):
        return np.array([np.sum(self.names_to_shapes[name]) for name in names])

    def dim(self, names):
        return np.sum(self.dims(names))
