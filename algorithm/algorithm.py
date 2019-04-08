from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import pickle
from torch.utils.data.dataloader import default_collate


class Algorithm(ABC):

    def __init__(self, data_files, value_function, policy):
        """
        Args:
            data_files (List): List of string of path
        """
        self.value_function = value_function
        self.policy = policy
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
        :param value_function
        :return: [s_i, reward, s_primes]
        """
        return batch[:self.value_function.forward_arg_size], batch[self.value_function.forward_arg_size], []

    @abstractmethod
    def _preprocess_file(self, file_data):
        """ Abstract method that should be implemented in the subclass. It must return a dictionary of dataset name (key),
        and the data contained in each dataset (value).

        :param file_data:
        :return:
        """
        pass

    def collate_func(self, batch):
        return self.deformat(default_collate(batch))

    def operator(self, values, idx_choosen):
        """ Methods to return the bootstrapping value that will be added to the reward.

        You can use the value_function and the policy to calculate this reward.

        The operator could result in a Sarsa, QLearning, or Expected Sarsa RL algorithm.

        :param values: values to select from
        :param values: index of the value that was choose during play.
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