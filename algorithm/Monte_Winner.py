from .algorithm import Algorithm
from .register import register
import numpy as np


@register
class WinnerPhase0(Algorithm):
    def __init__(self, data_files):
        super().__init__(data_files)

    def _preprocess_file(self, file_data):
        data = []

        for hand, hand_data in file_data['data'][0].items():

            sample = hand_data[0][0][0][0][hand_data[0][0][1]]
            data.append([sample, np.float32(file_data['winner'])])

        return {'Dataset': data}


@register
class WinnerPhase1(Algorithm):
    def __init__(self, data_files):
        super().__init__(data_files)

    def _preprocess_file(self, file_data):
        datasets = {}
        for hand, hand_data in file_data['data'][1].items():

            for i, ((s_i, idx_choice), R_i_plus_1) in enumerate(file_data['data'][1][hand]):

                s_prime = []

                s_i_choice = [n[idx_choice] for n in s_i]

                # Resolve dataset (must be 1 dataset by state sequence length and state prime sequence length pair)
                seq_len_s_i_choice = s_i_choice[0].shape[0]

                seq_len_s_prime = s_prime[0].shape[0] if len(s_prime) > 0 else 0
                data_set_name = 'i='+str(seq_len_s_i_choice)+'prime='+str(seq_len_s_prime)
                if data_set_name in datasets:
                    datasets[data_set_name].append(s_i_choice+[np.float32(file_data['winner'])]+s_prime)
                else:
                    datasets[data_set_name] = [s_i_choice + [np.float32(file_data['winner'])] + s_prime]

        return datasets

    def deformat(self, batch):
        return batch[:2], batch[2], batch[3:]