class Visualizer(object):

    def __init__(self, params, env_spec):
        self._env_spec = env_spec
        self._init_params_to_attrs(params)
        self._init_setup()

    def _init_params_to_attrs(self, params):
        pass

    def _init_setup(self):
        pass

    def step(self, model, observation, goal, done, get_action):
        pass
