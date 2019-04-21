from .algorithm import Algorithm
from .register import register
import numpy as np
from torch.utils.data.dataloader import default_collate
import torch
from ..utils.device import device

@register
class AllRewards_Phase0(Algorithm):
    def __init__(self, data_files, value_function, policy):
        super().__init__(data_files, value_function, policy)

    def _preprocess_file(self, file_data):
        data = []

        for hand, hand_data in file_data['data'][0].items():
            reward_phase0 = hand_data[0][1]['raw']

            if hand in file_data['data'][1]:
                reward_phase1 = sum([s[1]['raw'] for s in file_data['data'][1][hand]])

                if hand in file_data['data'][2]:
                    reward_phase2 = file_data['data'][2][hand][0][1]['raw']

                    sample = []
                    for s in hand_data[0][0][0]:
                        sample.append(s[hand_data[0][0][1]])

                    # If all phase are there for current hand we store data
                    G = reward_phase0+reward_phase1+reward_phase2
                    data.append(sample+[np.float32(G)])

        return {'Dataset': data}


@register
class NStep_Phase1(Algorithm):
    def __init__(self, data_files, value_function, policy, n_step=1, reward_data='raw'):
        self.n_step = n_step
        self.reward_data = reward_data
        super().__init__(data_files, value_function, policy)

    def _preprocess_file(self, file_data):
        data = []
        for hand, hand_data in file_data['data'][1].items():

            for i, ((s_i, idx_choice), R_i_plus_1) in enumerate(hand_data):

                R_i_plus_1 = R_i_plus_1[self.reward_data]
                boot_strap_idx = len(hand_data) if self.n_step is None else min(i+self.n_step, len(hand_data))

                for j in range(i+1, boot_strap_idx):
                    R_i_plus_1 += hand_data[j][1][self.reward_data]

                s_prime = [None, None]
                if boot_strap_idx < len(hand_data):
                    s_prime = hand_data[boot_strap_idx][0]

                s_i_choice = [n[idx_choice] for n in s_i]

                data.append([s_i_choice, np.float32(R_i_plus_1), s_prime])

        return {'Dataset': data}

    def collate_func(self, batch):
        # Deformat data batch produced by Pytorch dataloader
        s_i_reward_batch = []
        s_prime = []
        for sample in batch:
            s_i_reward_batch.append([*sample[0], sample[1]])
            if sample[2][0] is not None:
                s_prime.append([[torch.tensor(i) for i in sample[2][0]], sample[2][1]])
            else:
                s_prime.append(sample[2])

        s_i, reward = self.deformat(default_collate(s_i_reward_batch))[:-1]
        return s_i, reward, s_prime

@register
class NStep_Sarsa(NStep_Phase1):

    def operator(self, values, idx_choosen):
        return values[idx_choosen]

@register
class NStep_QLearning(NStep_Phase1):

    def operator(self, values, idx_choosen):
        return values.max()

@register
class NStep_ExpectedSarsa(NStep_Phase1):

    def operator(self, values, idx_choosen):
        prob = device(self.policy.get_transition_probabilities(values))
        return (values*prob).sum()