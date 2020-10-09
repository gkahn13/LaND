import argparse
import os

from sidewalk.experiments.file_manager import FileManager
from sidewalk.utils.file_utils import import_config
from sidewalk.utils.tf_utils import enable_eager_execution

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--continue', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_frac', type=float, default=0.3)
args = parser.parse_args()

enable_eager_execution(gpu=args.gpu, gpu_frac=args.gpu_frac)

# import the config params
config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
params = import_config(config_fname)
params.freeze()

file_manager = FileManager(params.exp_name,
                           is_continue=getattr(args, 'continue'),
                           log_fname='log_train.txt',
                           config_fname=config_fname)

# instantiate classes from the params
env_spec = params.env_spec.cls(params.env_spec.params)
dataset_train = params.dataset_train.cls(params.dataset_train.params, env_spec)
dataset_holdout = params.dataset_holdout.cls(params.dataset_holdout.params, env_spec)
model = params.model.cls(params.model.params, env_spec)
trainer = params.trainer.cls(params.trainer.params,
                             env_spec=env_spec,
                             file_manager=file_manager,
                             model=model,
                             dataset_train=dataset_train,
                             dataset_holdout=dataset_holdout)

# run training
trainer.run()