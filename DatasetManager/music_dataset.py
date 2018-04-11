from abc import ABC, abstractmethod
import os
from torch.utils.data import TensorDataset, DataLoader
import torch


class MusicDataset(ABC):
    """
    Abstract Base Class for music data sets
    """

    def __init__(self, cache_dir):
        self._tensor_dataset = None
        self.cache_dir = cache_dir

    @abstractmethod
    def make_tensor_dataset(self):
        """

        :return: TensorDataset
        """
        pass

    @property
    def tensor_dataset(self):
        """
        Loads or computes TensorDataset
        :return: TensorDataset
        """
        if self._tensor_dataset is None:
            if self.tensor_dataset_is_cached():
                print(f'Loading TensorDataset for {self.__repr__()}')
                self._tensor_dataset = torch.load(self.tensor_dataset_filepath)
            else:
                print(f'Creating {self.__repr__()} TensorDataset'
                      f' since it is not cached')
                self._tensor_dataset = self.make_tensor_dataset()
                torch.save(self._tensor_dataset, self.tensor_dataset_filepath)
                print(f'TensorDataset for {self.__repr__()} '
                      f'saved in {self.tensor_dataset_filepath}')
        return self._tensor_dataset

    @tensor_dataset.setter
    def tensor_dataset(self, value):
        self._tensor_dataset = value

    def tensor_dataset_is_cached(self):
        return os.path.exists(self.tensor_dataset_filepath)

    @property
    def tensor_dataset_filepath(self):
        tensor_datasets_cache_dir = os.path.join(
            self.cache_dir,
            'tensor_datasets')
        if not os.path.exists(tensor_datasets_cache_dir):
            os.mkdir(tensor_datasets_cache_dir)
        fp = os.path.join(
            tensor_datasets_cache_dir,
            self.__repr__()
        )
        return fp

    @property
    def filepath(self):
        tensor_datasets_cache_dir = os.path.join(
            self.cache_dir,
            'tensor_datasets')
        if not os.path.exists(tensor_datasets_cache_dir):
            os.mkdir(tensor_datasets_cache_dir)
        return os.path.join(
            self.cache_dir,
            'datasets',
            self.__repr__()
        )

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        """
        Returns three data loaders obtained by splitting
        self.tensor_dataset according to split
        :param batch_size:
        :param split:
        :return:
        """
        assert sum(split) < 1

        dataset = self.tensor_dataset
        num_examples = dataset.data_tensor.size()[0]
        a, b = split
        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset = TensorDataset(*dataset[int(a * num_examples):
                                             int((a + b) * num_examples)])
        eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl
