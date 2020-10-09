import numpy as np

from sidewalk.envs.env_spec import EnvSpec
from sidewalk.utils import abstract


class JackalEnvSpec(EnvSpec):

    def __init__(self, params):
        super().__init__(
            params,
            names_shapes_limits_dtypes=(
                ('images/front', (96, 192, 3), (0, 255), np.uint8),
                ('collision', (1,), (0, 1), np.float32),

                ('commands/turn', (1,), (-0.5, 0.5), np.float32),
                ('commands/dt', (1,), (0., 1.0), np.float32)
            )
        )

        self._observation_names = params.get('observation_names', ('images/front', 'collision'))

    @property
    @abstract.overrides
    def observation_names(self):
        return self._observation_names

    @property
    @abstract.overrides
    def output_observation_names(self):
        return ('collision',)

    @property
    @abstract.overrides
    def action_names(self):
        return ('commands/turn',)
