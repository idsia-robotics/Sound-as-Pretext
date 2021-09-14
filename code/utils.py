import torch
import numpy as np
from tf.transformations import quaternion_from_euler, translation_matrix, quaternion_matrix, decompose_matrix


def transf_matrix(pose):
    '''Converts a SE(3) pose stored in XYZQuat format to the respective homogeneous transformation matrix.

    Args:
            pose: a XYZQuat array-like pose.

    Returns:
            the 4x4 homogeneous transformation matrix.
    '''
    t = translation_matrix(pose[:3])
    r = quaternion_matrix(pose[3:7])
    return np.dot(t, r)


def se3_pose(transf_matrix):
    '''Converts a homogeneous transformation matrix to the respective SE(3) pose stored in XYZQuat format.

    Args:
            transf_matrix: a 4x4 homogeneous transformation matrix.

    Returns:
            the XYZQuat ndarray pose.
    '''
    _, _, rpy, xyz, _ = decompose_matrix(transf_matrix)
    rpy = quaternion_from_euler(*rpy, axes='sxyz')
    rpy /= np.linalg.norm(rpy)
    return np.array(xyz.tolist() + rpy.tolist())


def fliplr(data, p=.5):
    '''Flips horizontally the data dict with a given probability.

    Args:
            data: a dict of numpy arrays.
            p: the probability of flipping horizontally data.

    Returns:
            the data dict with image, position and sound features flipped horizontally.
    '''
    if np.random.rand() < p:
        im = data['rm_s1_camera_image_h264']
        im = np.array([i[..., ::-1] for i in im])
        data['rm_s1_camera_image_h264'] = im

        if 'optitrack_tello_proj' in data:
            y = data['optitrack_tello_proj']
            y[..., 0] = 1 - y[..., 0]
            data['optitrack_tello_proj'] = y

        if 'mp3_features' in data:
            feat = data['mp3_features']

            temp = feat[..., 0:3].copy()
            feat[..., 0:3] = feat[..., 3:6]
            feat[..., 3:6] = temp
            feat[..., 6:9] = feat[..., 0:3] - feat[..., 3:6]

            data['mp3_features'] = feat

    return data


def apply_albumentations(data, transform, index):
    '''Applies albumentations on the image of the data dict.

    Args:
            data: a dict of numpy arrays.
            transform: a function composition of albumnetations to be applied on x.
            index: index within dict of the images.

    Returns:
            the data dict with the image augmented.
    '''
    im = data[index]
    is_batched = im.ndim == 4

    if is_batched:
        for i in range(im.shape[0]):
            im[i] = transform(image=im[i])['image']
    else:
        im = transform(image=im)['image']

    data[index] = im

    return data


def to_tensor(data, device='cpu', dtype=torch.float):
    '''Converts a data dict of numpy arrays to a dict of torch tensors.

    Args:
            data: a dict of numpy arrays.
            device: device on which to perform computations (usually "cuda" or "cpu").
            dtype: a torch numeric type.

    Returns:
            the data dict of torch tensors.
    '''
    def internal(x):
        x = np.ascontiguousarray(x)
        x = torch.tensor(x, device=device, dtype=dtype)
        return x

    return {k: internal(v) for k, v in data.items()}
