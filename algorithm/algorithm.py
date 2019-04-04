from torch.utils.data import Dataset
import pickle
from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self, data_files):
        """
        Args:
            data_files (List): List of string of path
        """
        self.datasets = self._preprocess_data(data_files)

    def _preprocess_data(self, data_files):
        """ Preprocess all given data files and creates a dictionary of datasets.
        Datasets are created by the _preprocess_file abstract method in the subclass. Data is extend to the dataset for
        every file (multiple files forms multiple datasets).

        :param data_files:
        :return:
        """
        datasets = {}
        for data_file in data_files:
            file_data = pickle.load(open(data_file, 'rb'))
            for dataset_name, data in self._preprocess_file(file_data).items():
                if dataset_name in datasets:
                    datasets[dataset_name].extend(data)
                else:
                    datasets[dataset_name] = DatasetWrapper(data)

        return datasets

    def deformat(self, batch):
        """ Deformat batch in a list of [s_i, reward, *s_primes]

        s_i: List of the current state of where the gradient will be calculated.
        reward: Reward.
        s_primes: List of states (formatted like s_i) if any.

        By default, the value returned by the dataloader is a list of 2 elements with s_i (not in a list) and the
        reward. Otherwise, this method as to be overloaded in the subclass.

        :param batch:
        :return: [s_i, reward, s_primes]
        """
        return [batch[0]], batch[1], []

    @abstractmethod
    def _preprocess_file(self, file_data):
        """ Abstract method that should be implemented in the subclass. It must return a dictionary of dataset name (key),
        and the data contained in each dataset (value).

        :param file_data:
        :return:
        """
        pass


class DatasetWrapper(Dataset):

    def __init__(self, data):
        self.data = data

    def extend(self, data):
        self.data.extend(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]