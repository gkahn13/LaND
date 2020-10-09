import argparse
import os

from sidewalk.experiments.file_manager import FileManager
from sidewalk.utils.file_utils import import_config
from sidewalk.utils.python_utils import exit_on_ctrl_c
from sidewalk.utils.tf_utils import enable_eager_execution, enable_static_execution

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--eager', type=int, default=False)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_frac', type=float, default=0.3)
args = parser.parse_args()

if args.eager:
    enable_eager_execution(gpu=args.gpu, gpu_frac=args.gpu_frac)
else:
    enable_static_execution(gpu=args.gpu, gpu_frac=args.gpu_frac)

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)

params = import_config(config_fname)
params.freeze()

file_manager = FileManager(params.exp_name, is_continue=True)
env_spec = params.env_spec.cls(params.env_spec.params)
env = params.env.cls(params.env.params, env_spec)
model = params.model.cls(params.model.params, env_spec)
policy = params.policy.cls(params.policy.params, env_spec)
visualizer = params.visualizer.cls(params.visualizer.params, env_spec)

### warm start the planner
obs, goal = env.reset()
policy.warm_start(model, obs, goal)

### restore model
model.restore(args.model)

### eval loop
exit_on_ctrl_c()
done = True
while True:
    if done:
        obs, goal = env.reset()

    get_action = policy.get_action(model, obs, goal)
    visualizer.step(model, obs, goal, done, get_action)
    obs, goal, done = env.step(get_action.action)
