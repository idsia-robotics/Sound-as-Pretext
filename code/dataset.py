import h5py
import torch
import numpy as np
import albumentations as A
from torchvision.transforms import Compose
from utils import apply_albumentations, to_tensor, fliplr


def get_dataset(filename, split, augment, device):
    """Returns a dataset of instances.

    Args:
            filename: a filename of an HDF5 file.
            split: one of ("t1", "t2", "val" or "test").
            augment: if set applies data augmentation.
            device: device on which to perform computations (usually "cuda" or "cpu").

    Note:
            t1 = unlabeled training-set
            t2 = labeled training-set

    Returns:
            the dataset of instances.
    """
    transforms = []

    if augment:
        transforms.append(lambda data:
                          apply_albumentations(data, A.Compose([
                              A.Blur(blur_limit=3, p=.5),
                              A.GaussNoise(.002, p=.5),
                              A.RandomBrightnessContrast(.1, .1, p=.5),
                              A.ToFloat()
                          ]), index='rm_s1_camera_image_h264'))

        transforms.append(fliplr)

    transforms += [
        lambda data: to_tensor(data, device=device)
    ]

    return PretextDataset(filename, split, Compose(transforms))


class PretextDataset():

    def __init__(self, filename, split, transform=lambda x: x):
        self.split = split
        self.transform = transform

        if split not in ['t1', 't2', 'val', 'test']:
            raise ValueError(
                self.split + ' is not a valid split, use one of t1, t2, val, or test')

        prefix = '21_rosbag2_2021_07_09-11_14_04' if split == 'test' else 'virtual'
        self.h5f = h5py.File(filename + '_' + split + '.h5', 'r')
        self.start = 0
        self.end = self.h5f[prefix + '/rm_s1_camera_image_h264'].shape[0]
        self.data = self.h5f[prefix]

    def __len__(self):
        return self.end - self.start

    def __del__(self):
        self.h5f.close()

    def __getitem__(self, slice):
        if isinstance(slice, int):
            slice = self._process_index(slice, self.start)
        else:
            slice = self._process_slice(slice.start, slice.stop, slice.step)

        data = {str(k): v[slice] for k, v in self.data.items()}

        data = self.transform(data)

        return data

    def _process_index(self, index, default):
        if index is None:
            index = default
        elif index < 0:
            index += self.end
        else:
            index += self.start

        return index

    def _process_slice(self, start=None, stop=None, step=None):
        return slice(self._process_index(start, self.start),
                     self._process_index(stop, self.end),
                     step)

    def batches(self, batch_size, shuffle=False):
        length = len(self)
        indices = np.arange(0, length, batch_size)

        if shuffle:
            indices = np.random.permutation(indices)

        for start in indices:
            end = min(start + batch_size, length)
            yield self[start:end]
