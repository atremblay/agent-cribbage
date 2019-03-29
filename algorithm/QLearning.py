from .algorithm import Algorithm
from .register import register
import numpy as np

@register
class QLearningPhase0(Algorithm):
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
                    sample = hand_data[0][0][0][0][hand_data[0][0][1]]
                    data.append([sample, np.float32(G)])

        return data
