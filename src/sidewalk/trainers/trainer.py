from collections import defaultdict
import numpy as np
import tensorflow as tf

from sidewalk.experiments import logger
from sidewalk.utils.python_utils import timeit


class Trainer(object):

    def __init__(self, params, env_spec, file_manager, model, dataset_train, dataset_holdout):
        self._env_spec = env_spec
        self._file_manager = file_manager
        self._model = model
        self._dataset_train = dataset_train
        self._dataset_holdout= dataset_holdout

        # steps
        self._max_steps = int(params.max_steps)
        self._holdout_every_n_steps = int(params.holdout_every_n_steps)
        self._log_every_n_steps = int(params.log_every_n_steps)
        self._save_every_n_steps = int(params.save_every_n_steps)

        # cost
        self._cost_fn = params.cost_fn

        # optimizer
        optimizer_cls = params.optimizer_cls
        learning_rate = params.learning_rate

        # create optimizer and checkpoint
        self._optimizer = optimizer_cls(learning_rate)
        self._global_step = tf.train.get_or_create_global_step()
        self._checkpointer = tf.train.Checkpoint(optimizer=self._optimizer,
                                                 model=model,
                                                 optimizer_step=self._global_step)

        # tensorboard logging
        self._tb_writer = tf.contrib.summary.create_file_writer(file_manager.exp_dir,
                                                                max_queue=100,
                                                                flush_millis=5000)
        self._tb_writer.set_as_default()
        self._tb_logger = defaultdict(list)

    def run(self):
        self._restore_latest_checkpoint()

        for step in range(self._get_current_step(), self._max_steps + 1):
            with timeit('total'):
                self._train_step()

                if step > 0 and step % self._holdout_every_n_steps == 0:
                    with timeit('holdout'):
                        self._holdout_step()

                if step > 0 and step % self._save_every_n_steps == 0:
                    with timeit('save'):
                        self._save()

            if step > 0 and step % self._log_every_n_steps == 0:
                self._log()

    def _restore_latest_checkpoint(self):
        # restore checkpoint
        latest_ckpt_fname = tf.train.latest_checkpoint(self._file_manager.ckpts_dir)
        if latest_ckpt_fname:
            logger.info('Restoring ckpt {0}'.format(latest_ckpt_fname))
            self._checkpointer.restore(latest_ckpt_fname)
            logger.info('Starting training from step = {0}'.format(self._get_current_step()))

    def _get_current_step(self):
        return self._global_step.numpy()

    def _train_step(self):
        with timeit('train_batch'):
            inputs, outputs = self._dataset_train.get_batch()

        with timeit('train_update'):
            with tf.GradientTape() as tape:
                model_outputs = self._model.call(inputs, training=True)
                cost = self._cost_fn(inputs, outputs, model_outputs, self._env_spec)
                grads = tape.gradient(tf.reduce_mean(cost.total), self._model.trainable_variables)

            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables), self._global_step)

            for name, tensor in cost.items():
                self._tb_logger['train_cost_' + name] += tensor.numpy().tolist()

    def _holdout_step(self):
        inputs, outputs = self._dataset_holdout.get_batch()
        model_outputs = self._model.call(inputs, training=True)
        cost = self._cost_fn(inputs, outputs, model_outputs, self._env_spec)

        for name, tensor in cost.items():
            self._tb_logger['holdout_cost_' + name] += tensor.numpy().tolist()

    def _save(self):
        logger.info('Saving model...')
        self._checkpointer.save(self._file_manager.ckpt_prefix)

    def _log(self):
        logger.info('')
        logger.info('Step {0}'.format(self._get_current_step() - 1))
        with tf.contrib.summary.always_record_summaries():
            for key, value in sorted(self._tb_logger.items(), key=lambda kv: kv[0]):
                logger.info('{0} {1:.6f}'.format(key, np.mean(value)))
                tf.contrib.summary.scalar(key, np.mean(value))
            self._tb_logger.clear()

        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()
