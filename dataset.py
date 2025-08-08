import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    def __init__(self, data_dir,num_classes=1000):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.num_classes = num_classes
        self.images_dir = os.path.join(data_dir, 'images')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.images_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.image_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        return torch.from_numpy(image), torch.tensor(self.labels[idx])


class MockDataset(Dataset):
    def __init__(self, num_samples=100, image_shape=(3, 256, 256), num_classes=1000):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.labels = np.random.randint(0, num_classes, size=(num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = np.random.rand(*self.image_shape).astype(np.float32)
        label = self.labels[idx]
        return torch.from_numpy(image), torch.tensor(label)