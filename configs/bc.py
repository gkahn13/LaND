import numpy as np
import os
import tensorflow as tf

from sidewalk.experiments.file_manager import FileManager
from sidewalk.policies.static_cem import StaticCEMPolicy
from sidewalk.robots.jackal.datasets.jackal_hdf5_dataset import JackalHdf5Dataset
from sidewalk.robots.jackal.models.jackal_imitation_model import JackalImitationModel
from sidewalk.robots.jackal.envs.jackal_env_spec import JackalEnvSpec
from sidewalk.robots.jackal.envs.jackal_hdf5_env import JackalHdf5Env
from sidewalk.robots.jackal.visualizers.jackal_imitation_visualizer import JackalImitationVisualizer
from sidewalk.trainers.trainer import Trainer
from sidewalk.utils.python_utils import AttrDict as d


def get_env_spec_params():
    return d(
        cls=JackalEnvSpec,
        params=d(
        )
    )

def get_env_params():
    return d(
        cls=JackalHdf5Env,
        params=d(
            hdf5_folders=[os.path.join(FileManager.experiments_dir, 'hdf5s/train')],
        )
    )

def get_dataset_params(horizon, folders):
    return d(
        cls=JackalHdf5Dataset,
        params=d(
            batch_size=32,
            horizon=horizon,

            hdf5_folders=folders,

            rebalance=False,
            truncate_each_rollout_by=4,
        )
    )

def get_model_params(horizon):
    return d(
        cls=JackalImitationModel,
        params=d(
            observation_im_names=[
                'images/front',
            ],
            observation_vec_names=[
            ],

            finetune_cnn=True,
            init_with_imagenet_weights=True,

            bin_edges=np.deg2rad(np.r_[-60.:61.:5.]),
        )
    )

def get_trainer_params():

    def cost_fn(inputs, outputs, model_outputs, env_spec):
        batch_size = outputs.done.shape.as_list()[0]

        # turn
        cost_turn = -model_outputs.log_prob
        accuracy_turn = model_outputs.acc

        # regularization
        cost_l2_reg = 1e-2 * \
                      tf.reduce_mean([0.5 * tf.reduce_mean(kernel * kernel) for kernel in model_outputs.kernels]) * \
                      tf.ones(batch_size)

        cost = cost_turn + cost_l2_reg

        d_cost= d(
            total=cost,
            turn=cost_turn,
            turn_accuracy=accuracy_turn,
            l2_reg=cost_l2_reg
        )

        return d_cost

    return d(
        cls=Trainer,
        params=d(
            max_steps=int(1e5),
            holdout_every_n_steps=50,
            log_every_n_steps=int(1e2),
            save_every_n_steps=int(1e4),

            cost_fn=cost_fn,

            optimizer_cls=tf.train.AdamOptimizer,
            learning_rate=1e-4,
        )
    )

def get_policy_params(horizon):
    def cost_fn(inputs, model_outputs, goals, actions):
        ### probability
        prob_cost = 1. - tf.exp(model_outputs.log_prob[..., tf.newaxis])

        ### action
        turn = actions.commands.turn[..., 0]
        turn_cost = 0.5 * tf.square(turn)

        ### total
        total = prob_cost + 0.1 * turn_cost

        return d(
            total=total,
            prob_cost=prob_cost
        ) # [batch, horizon]

    return d(
        cls=StaticCEMPolicy,
        params=d(
            horizon=horizon,
            action_selection_limits=d(
                commands=d(
                    turn=(-0.5, 0.5),
                )
            ),
            cost_fn=cost_fn,

            # CEM params
            M_init=8192,
            M=4096,
            K=512,
            itrs=3,
            eps=1e-3,
        )
    )

def get_visualizer_params():
    return d(
        cls=JackalImitationVisualizer,
        params=d(
        )
    )


def get_params():
    horizon = 1

    train_folders = [os.path.join(FileManager.experiments_dir, 'hdf5s/train')]
    holdout_folders = [os.path.join(FileManager.experiments_dir, 'hdf5s/holdout')]

    return d(
        exp_name='bc',

        # NOTE: this is where all the params get created
        env_spec=get_env_spec_params(),
        env=get_env_params(),
        dataset_train=get_dataset_params(horizon, train_folders),
        dataset_holdout=get_dataset_params(horizon, holdout_folders),
        model=get_model_params(horizon),
        trainer=get_trainer_params(),
        policy=get_policy_params(horizon),
        visualizer=get_visualizer_params()
    )

# NOTE: this params is what will be imported
params = get_params()
