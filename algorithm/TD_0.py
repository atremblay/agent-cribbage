from .algorithm import Algorithm
from .register import register
import numpy as np


@register
class TD0Phase0(Algorithm):
    def __init__(self, data_files):
        super().__init__(data_files)

    def _preprocess_file(self, file_data):
        data = []

        for hand, hand_data in file_data['data'][0].items():
            reward_phase0 = hand_data[0][1]

            if hand in file_data['data'][1]:
                reward_phase1 = sum([s[1] for s in file_data['data'][1][hand]])

                if hand in file_data['data'][2]:
                    reward_phase2 = file_data['data'][2][hand][0][1]

                    # If all phase are there for current hand we store data
                    G = reward_phase0+reward_phase1+reward_phase2
                    sample = hand_data[0][0][0][0][hand_data[0][0][1][0]]
                    data.append([sample, np.float32(G)])

        return {'Dataset': data}


@register
class TD0Phase1(Algorithm):
    def __init__(self, data_files, callback):
        super().__init__(data_files)
        self.callback = getattr(self, callback)

    def _preprocess_file(self, file_data):
        datasets = {}
        for hand, hand_data in file_data['data'][1].items():

            for i, ((s_i, (idx_choice, prob)), R_i_plus_1) in enumerate(file_data['data'][1][hand]):

                s_i_plus_1 = []
                if i < len(hand_data)-1:
                    s_i_plus_1, (idx_choice_plus_1, prob) = hand_data[i+1][0]

                s_i_choice = [n[idx_choice] for n in s_i]

                # Resolve dataset (must be 1 dataset by state sequence length and state prime sequence length pair)
                seq_len_s_i_choice = s_i_choice[0].shape[0]

                seq_len_s_prime_dim0 = s_i_plus_1[0].shape[0] if len(s_i_plus_1) > 0 else np.float32(0)
                seq_len_s_prime_dim1 = s_i_plus_1[0].shape[1] if len(s_i_plus_1) > 0 else np.float32(0)
                data_set_name = 'i='+str(seq_len_s_i_choice)+'prime1='+str(seq_len_s_prime_dim0)+'prime1='+str(seq_len_s_prime_dim1)
                if data_set_name in datasets:
                    datasets[data_set_name].append(s_i_choice+[np.float32(R_i_plus_1)]+s_i_plus_1+[np.array(prob, dtype=np.float32)])
                else:
                    datasets[data_set_name] = [s_i_choice + [np.float32(R_i_plus_1)] + s_i_plus_1 + [np.array(prob, dtype=np.float32)]]

        return datasets

    def deformat(self, batch):
        return batch[:2], batch[2], batch[3:]

    def Expected(self, values, prob, importance_sampling_ratio, idx):
        return (importance_sampling_ratio*values*prob).sum()

    def TD(self, values, prob, importance_sampling_ratio, idx):
        return importance_sampling_ratio[idx] * values[idx]