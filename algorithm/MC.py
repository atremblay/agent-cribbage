from .algorithm import Algorithm
from .register import register
import numpy as np


@register
class MCPhase1(Algorithm):
    def __init__(self, data_files, ratio_denominator=1):
        self.ratio_denominator = ratio_denominator
        super().__init__(data_files)

    def _preprocess_file(self, file_data):
        datasets = {'dataset': []}
        for hand, hand_data in file_data['data'][1].items():

            for i, ((s_i, idx_choice), R_i_plus_1) in enumerate(file_data['data'][1][hand]):

                s_i_choice = [n[idx_choice] for n in s_i]

                G = sum(rewards)

                datasets['dataset'].append(s_i_choice+[np.float32(G)])

        return datasets