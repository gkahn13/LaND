import cv2
import io
from math import factorial
import numpy as np
import os
import PIL, PIL.Image
try:
    from scipy.signal import butter, lfilter
    import skvideo.io
except:
    print('!!!!!!! not importing scipy/skvideo')
import tempfile


def imrectify_fisheye(img, K, D, balance=0.0):
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    dim = img.shape[:2][::-1]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def imresize(image, shape, resize_method=PIL.Image.LANCZOS):
    if len(image.shape) == 4:
        return np.stack([imresize(image_i, shape, resize_method=resize_method) for image_i in image])

    assert (len(shape) == 3)
    assert (shape[-1] == 1 or shape[-1] == 3)
    assert (image.shape[0] / image.shape[1] == shape[0] / shape[1]) # maintain aspect ratio
    height, width, channels = shape

    if len(image.shape) > 2 and image.shape[2] == 1:
        image = image[:,:,0]

    im = PIL.Image.fromarray(image)
    im = im.resize((width, height), resize_method)
    im = np.array(im)

    if len(im.shape) == 2:
        im = np.expand_dims(im, 2)

    assert (im.shape == tuple(shape))

    return im


def im2bytes(arrs, format='jpg', quality=75):
    if len(arrs.shape) == 4:
        return np.array([im2bytes(arr_i, format=format, quality=quality) for arr_i in arrs])
    elif len(arrs.shape) == 3:
        if arrs.shape[-1] == 1:
            arrs = arrs[..., 0]
        im = PIL.Image.fromarray(arrs.astype(np.uint8))
        with io.BytesIO() as output:
            im.save(output, format="jpeg", quality=quality)
            return output.getvalue()
    else:
        raise ValueError


def bytes2im(arrs):
    if len(arrs.shape) == 1:
        return np.array([bytes2im(arr_i) for arr_i in arrs])
    elif len(arrs.shape) == 0:
        return np.array(PIL.Image.open(io.BytesIO(arrs)))
    else:
        raise ValueError


def compress_video(images, crf=30):
    _, fname = tempfile.mkstemp(suffix='.mp4')
    writer = skvideo.io.FFmpegWriter(fname, outputdict={'-vcodec': 'libx264', '-crf': str(crf)})
    for im in images:
        writer.writeFrame(im)
    writer.close()

    with open(fname, 'rb') as f:
        bytes = f.read()

    os.remove(fname)
    return np.frombuffer(bytes, dtype=np.uint8)


def uncompress_video(buffer):
    _, fname = tempfile.mkstemp(suffix='.mp4')
    with open(fname, 'wb') as f:
        f.write(buffer)

    for frame in skvideo.io.vreader(fname):
        yield frame

    os.remove(fname)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    #https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def yaw_rotmat(yaw):
    dims = list(yaw.shape)
    return np.reshape(
        np.stack([np.cos(yaw), -np.sin(yaw),
                  np.sin(yaw), np.cos(yaw)],
                 axis=-1),
        dims + [2, 2]
    )
