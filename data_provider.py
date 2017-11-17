from data_utils import load_CIFAR10
import numpy as np

cifar10_dir = 'cifar-10-batches-py'


class DataProvider(object):
    def __init__(self):
        self._train_data, self._train_label, self._test_data, self._test_label = load_CIFAR10(cifar10_dir)

    def get_data(self, phase):
        train_size = int(len(self._train_data) * 2 / 3)
        if phase == 'train':
            return self._train_data[:train_size], self._train_label[:train_size]
        elif phase == 'val':
            return self._train_data[train_size:], self._train_label[train_size:]
        elif phase == 'trainval':
            return self._train_data, self._train_label
        elif phase == 'test':
            return self._test_data, self._test_label

    def get_cv_data(self, i):
        s = int(len(self._train_data) / 5)
        if i == 0:
            train_data = self._train_data[s:]
            train_label = self._train_label[s:]
            val_data = self._train_data[:s]
            val_label = self._train_label[:s]
        elif 0 < i < 4:
            train_data = np.concatenate((self._train_data[:(i*s)], self._train_data[((i+1)*s):]), axis=0)
            train_label = np.concatenate((self._train_label[:(i*s)], self._train_label[((i+1)*s):]), axis=0)
            val_data = self._train_data[(i*s):((i+1)*s)]
            val_label = self._train_label[(i*s):((i+1)*s)]
        elif i == 4:
            train_data = self._train_data[:(i*s)]
            train_label = self._train_label[:(i*s)]
            val_data = self._train_data[(i*s):]
            val_label = self._train_label[(i*s):]
        return train_data, train_label, val_data, val_label

    def next_batch(self, batch_size, phase):
        """

        :param batch_size:
        :param phase: train, val or test
        :return:
        """
        raise NotImplementedError

