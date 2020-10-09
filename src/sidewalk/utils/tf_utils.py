import os
import tensorflow as tf


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def config_gpu(gpu=0, gpu_frac=0.3):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    return config


def enable_eager_execution(gpu=0, gpu_frac=0.3):
    tf.enable_eager_execution(
        config=config_gpu(gpu=gpu, gpu_frac=gpu_frac),
        # device_policy=tf.contrib.eager.DEVICE_PLACEMENT_EXPLICIT
    )


def enable_static_execution(gpu=0, gpu_frac=0.3):
    graph = tf.Graph()
    session = tf.Session(graph=graph, config=config_gpu(gpu=gpu, gpu_frac=gpu_frac))
    session.__enter__() # so get default session works


def get_kernels(layers):
    kernels = []
    for layer in layers:
        if hasattr(layer, 'layers'):
            kernels += get_kernels(layer.layers)
        elif hasattr(layer, 'kernel'):
            kernels.append(layer.kernel)
    return kernels


def repeat(tensor, repeats):
    """
    https://github.com/tensorflow/tensorflow/issues/8246
    Args:
        input: A Tensor. 1-D or higher.
        repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
    Returns:
        A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)

    return repeated_tesnor


def yaw_rotmat(yaw):
    batch_size = tf.shape(yaw)[0]
    nonbatch_dims = yaw.shape.as_list()[1:]
    return tf.reshape(
        tf.stack([tf.cos(yaw), -tf.sin(yaw),
                  tf.sin(yaw), tf.cos(yaw)],
                 axis=-1),
        [batch_size] + nonbatch_dims + [2, 2]
    )
