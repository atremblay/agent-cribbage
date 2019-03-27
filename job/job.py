import argparse
from logger_toolbox import setup_logging
import os
import getpass
import sys
import logging


class Job:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('job', type=str)
        self.parser.add_argument('policy', choices=['Boltzmann', 'EpsilonGreedy', 'Random'])
        self.parser.add_argument('value_function0', choices=['FFW'])
        self.parser.add_argument('value_function1', choices=['LSTM', 'FFW'])
        self.parser.add_argument('--save', type=str, default='/home/execution')
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument("--tao", default=1, type=float)
        self.parser.add_argument("--epsilon", default=1e-3, type=float)
        self.parser.add_argument("--number_games", default=1, type=int)

    def setup_logging(self, name):
        self.args = self.parser.parse_args()
        self.args.save = os.path.join(self.args.save, getpass.getuser(), 'agent-cribbage', name)
        setup_logging(self.args.save)
        self.logger = logging.getLogger(name)
        # Append process Id that was created by logging
        self.args.save = os.path.join(self.args.save, 'pid' + str(os.getpid()))
        self.save_args()

    def __getitem__(self, item):
        return getattr(self.args, item)

    def __setitem__(self, item, value):
        return setattr(self.args, item, value)

    def save_args(self):
        with open(os.path.join(self.args.save, 'commandline_args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

    def template_agent_args(self):
        return [{'name': self['policy'], 'kwargs': self.get_policy_args()},
                {
                  'name0': self['value_function0'], 'kwargs0': self.get_value_function_args()[0],
                  'name1': self['value_function1'], 'kwargs1': self.get_value_function_args()[1]
                }]

    def get_policy_args(self):
        if self.args.policy == 'Boltzmann':
            return {'tao': self.args.tao}
        elif self.args.policy == 'EpsilonGreedy':
            return {'epsilon': self.args.epsilon}
        elif self.args.policy == 'Random':
            return {}

    def get_value_function_args(self):
        if self.args.value_function0 == 'FFW':
            VF0 = {}

        if self.args.value_function1 in ['LSTM', 'FFW']:
            VF1 = {}

        return VF0, VF1