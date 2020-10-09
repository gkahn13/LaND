import numpy as np
import tensorflow as tf

from sidewalk.utils import np_utils


def turns_and_steps_to_positions(turns, steps, is_tf=False):
    if is_tf:
        get_shape = lambda x: tuple(x.shape.as_list())
        module = tf
    else:
        get_shape = lambda x: tuple(x.shape)
        module = np

    assert get_shape(turns) == get_shape(steps)

    if len(turns.shape) == 1:
        is_batch = False
        turns = turns[module.newaxis]
        steps = steps[module.newaxis]
    elif len(turns.shape) == 2:
        is_batch = True
    else:
        raise ValueError

    batch_size, horizon = get_shape(turns)
    angles = [module.zeros(batch_size)]
    positions = [module.zeros((batch_size, 2))]
    for turn, step in zip(module.split(turns, horizon, axis=1), module.split(steps, horizon, axis=1)):
        turn = turn[:, 0]
        step = step[:, 0]

        angle = angles[-1] + turn
        position = positions[-1] + step[:, module.newaxis] * \
                   module.stack([module.cos(angle), module.sin(angle)], axis=-1)

        angles.append(angle)
        positions.append(position)
    positions = module.stack(positions, axis=1)

    if not is_batch:
        positions = positions[0]

    return positions


def positions_to_turns_and_steps(positions):
    turns, steps = [], []
    prev_angle = 0.
    prev_pos = np.array([0., 0.])
    for pos in positions:
        delta_pos = pos - prev_pos
        angle = np.arctan2(delta_pos[1], delta_pos[0])

        turn = angle - prev_angle
        step = np.linalg.norm(delta_pos)

        turns.append(turn)
        steps.append(step)

        prev_angle = angle
        prev_pos = pos

    return np.array(turns), np.array(steps)


def rectify_and_resize(image, shape, rectify=True):
    if rectify:
        ### jackal camera intrinsics
        fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 240.000000
        K = np.array([[fx, 0., cx],
                            [0., fy, cy],
                            [0., 0., 1.]])
        D = np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
        balance = 0.5

        if len(image.shape) == 4:
            return np.array([rectify_and_resize(im_i, shape) for im_i in image])

        image = np_utils.imrectify_fisheye(image, K, D, balance=balance)

    image = np_utils.imresize(image, shape)

    return image


def process_image(original_image, desired_shape, image_rectify):
    is_batch = len(original_image.shape) == 4
    if not is_batch:
        original_image = original_image[None]

    # get shapes and ratios
    original_shape = original_image.shape[-3:]
    original_ratio = original_shape[1] / float(original_shape[0])

    # resize
    resize_shape = np.array([desired_shape[1] / original_ratio, desired_shape[1], 3], dtype=int)
    original_image = rectify_and_resize(original_image, resize_shape, rectify=image_rectify)

    # crop
    crop = int(0.5 * (resize_shape[0] - desired_shape[0]))
    desired_image = original_image[:, crop:-crop]
    assert tuple(desired_image.shape[-3:]) == desired_shape

    if not is_batch:
        desired_image = desired_image[0]

    return desired_image
