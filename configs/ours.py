import numpy as np
import os
import tensorflow as tf

from sidewalk.experiments.file_manager import FileManager
from sidewalk.policies.static_mppi import StaticMPPIPolicy
from sidewalk.robots.jackal.datasets.jackal_hdf5_dataset import JackalHdf5Dataset
from sidewalk.robots.jackal.envs.jackal_env_spec import JackalEnvSpec
from sidewalk.robots.jackal.envs.jackal_hdf5_env import JackalHdf5Env
from sidewalk.robots.jackal.models.jackal_disengagement_model import JackalDisengagementModel
from sidewalk.robots.jackal.visualizers.jackal_disengagement_visualizer import JackalDisengagementVisualizer
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

def get_dataset_params(horizon, hdf5_folders):
    return d(
        cls=JackalHdf5Dataset,
        params=d(
            batch_size=32,
            horizon=horizon,

            hdf5_folders=hdf5_folders,
        )
    )

def get_model_params(horizon):
    return d(
        cls=JackalDisengagementModel,
        params=d(
            observation_im_names=[
                'images/front',
            ],
            observation_vec_names=[
            ],
            horizon=horizon,
            obs_lowd_dim=128,

            finetune_cnn=True,
            init_with_imagenet_weights=True,

            num_collision_bins=2
        )
    )

def get_trainer_params():

    def cost_fn(inputs, outputs, model_outputs, env_spec):
        batch_size = outputs.done.shape.as_list()[0]
        done = tf.concat([tf.zeros([batch_size, 1], dtype=tf.bool), outputs.done[:, :-1]], axis=1)

        ### collision
        model_output_collision = model_outputs.logits[..., 0]

        collision = tf.cast(outputs.collision, tf.bool)[..., 0]
        collision = tf.logical_and(collision, tf.logical_not(done))  # don't count collisions after done!
        collision = tf.cumsum(tf.cast(collision, tf.float32), axis=-1) > 0.5

        # collision mask should be same as normal mask, but turned on for dones with collision = true
        mask_collision = tf.cast(tf.logical_or(tf.logical_not(done), collision), tf.float32)
        mask_collision = float(batch_size) * (mask_collision / tf.reduce_sum(mask_collision))

        cost_collision = 2.0 * tf.reduce_sum(
            mask_collision * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(collision, tf.float32),
                                                                     logits=model_output_collision),
            axis=1
        )
        collision_accuracy = tf.reduce_mean(tf.cast(tf.equal(model_output_collision > 0,
                                                             tf.cast(collision, tf.bool)),
                                                    tf.float32),
                                            axis=1)
        collision_accuracy_random = tf.reduce_mean(1. - tf.cast(collision, tf.float32), axis=1)

        ### regularization

        cost_l2_reg = 1e-2 * \
                      tf.reduce_mean([0.5 * tf.reduce_mean(kernel * kernel) for kernel in model_outputs.kernels]) * \
                      tf.ones(batch_size)

        ### filter out nans

        costs_is_finite = tf.is_finite(cost_collision)
        cost_collision = tf.boolean_mask(cost_collision, costs_is_finite)
        cost_l2_reg = tf.boolean_mask(cost_l2_reg, costs_is_finite)

        ### total

        cost = cost_collision + cost_l2_reg

        return d(
            total=cost,
            collision=cost_collision,
            collision_accuracy=collision_accuracy,
            collision_accuracy_random=collision_accuracy_random,
            l2_reg=cost_l2_reg
        )

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
        ### discriminative
        discriminative_cost = model_outputs.probcoll[..., 0]
        gamma = 0.5
        weighting = np.power(gamma, np.arange(horizon))
        weighting *= horizon / weighting.sum()

        ### action
        turn = actions.commands.turn[..., 0]
        turn_cost = 0.5 * tf.square(turn)

        ### goal
        goal_turn_cost = goals.is_turn * tf.square(turn - goals.turn_goal)

        ### total
        total = 1. * discriminative_cost * weighting + 0.1 * turn_cost + 1. * goal_turn_cost

        return d(
            total=total,
            discriminative_cost=discriminative_cost
        ) # [batch, horizon]

    return d(
        cls=StaticMPPIPolicy,
        params=d(
            horizon=horizon,
            action_selection_limits=d(
                commands=d(
                    turn=(-0.5, 0.5),
                )
            ),
            cost_fn=cost_fn,

            # MPPI params
            sigma=1.0,
            N=8192,
            gamma=50.0,
            beta=0.5,
        )
    )

def get_visualizer_params():
    return d(
        cls=JackalDisengagementVisualizer,
        params=d(
        )
    )


def get_params():
    horizon = 4

    train_folders = [os.path.join(FileManager.experiments_dir, 'hdf5s/train')]
    holdout_folders = [os.path.join(FileManager.experiments_dir, 'hdf5s/holdout')]

    return d(
        exp_name='ours',

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