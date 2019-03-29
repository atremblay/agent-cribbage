from torch.utils.data import Dataset
import pickle


class Algorithm(Dataset):

    def __init__(self, data_files):
        """
        Args:
            data_files (List): List of string of path
        """
        self.data = self._preprocess_data(data_files)

    def _preprocess_data(self, data_files):
        data = []
        for data_file in data_files:
            test_data = pickle.load(open(data_file, 'rb'))
            data.extend(self._preprocess_file(test_data))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]