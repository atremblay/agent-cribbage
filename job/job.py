import argparse
import copy
import getpass
import logging
import os
import sys

import torch
import yaml

from ..agent.agent import Agent
from ..logger_toolbox import setup_logging
from ..utils.device import device


class Job:
    def __init__(self, agent=None):

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('job', type=str)
        self.parser.add_argument('--agent_yaml', type=str, default=None)
        self.parser.add_argument('--cuda', default=0, type=int, help="Cuda device to use (-1 = Cuda disabled)")
        self.parser.add_argument('--save', type=str, default='/home/execution')
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument("--number_games", default=1, type=int)
        self.agents = agent
        self.cuda = False

    def _setup_job(self, name, args, logger):
        if args is None:
            self.add_argument()  # Mandatory method in subclass
            self._setup_args(name)
        else:
            self.args = args

        if logger is None:
            self._setup_logging(name)
        else:
            self.logger = logger

        self.resolve_cuda()
        self.init_agent(self['agent_yaml'])

    def _setup_args(self, name):
        self.args = self.parser.parse_args()
        self.args.save = os.path.join(self.args.save, getpass.getuser(), 'agent-cribbage', name)
        # Append process Id that was created by logging
        self.args.save = os.path.join(self.args.save, 'pid' + str(os.getpid()))
        self.save_args()

    def _setup_logging(self, name):
        setup_logging(self.args.save, configFilePath='./agent-cribbage/logger_toolbox/logging.yaml')
        self.logger = logging.getLogger(name)

    def __getitem__(self, item):
        return getattr(self.args, item)

    def __setitem__(self, item, value):
        return setattr(self.args, item, value)

    def save_args(self):
        # Make directories
        try:
            os.makedirs(self.args.save)
        except FileExistsError:  # Problem with race condition
            pass

        with open(os.path.join(self.args.save, 'commandline_args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

    def init_agent(self, agent_yaml):
        """ Initialize agents

        :return:
        """

        if self.agents is None and agent_yaml is not None:
            # Parse agent yaml file
            with open(agent_yaml, 'rt') as file:
                config = yaml.safe_load(file.read())

            # Create all agents form config
            self.agents = []
            for shared_agent in config['Agents']:
                agent = Agent(**shared_agent['kwargs'])

                # Cudaize models
                for v in agent.value_functions:
                    v.cuda(self['cuda'])

                if shared_agent['file'] is not None:
                    self['epoch_start'] = agent.load_checkpoint(shared_agent['file'])
                self.agents.extend(self.config_shared_agent(shared_agent['number'], agent))

        assert agent_yaml is None or len(self.agents) == 2  # Environment does not support more than 2 players now

    @staticmethod
    def config_shared_agent(number_of_shared_agent, agent):
        """ Configure multiple agent that share the same value function pointer

        :param number_of_shared_agent:
        :param agent:
        :return:
        """
        agents = [agent]
        for i_agent in range(1, number_of_shared_agent):
            agents.append(copy.deepcopy(agent))
            agents[-1].value_functions = agent.value_functions  # share the same value function

        return agents

    def resolve_cuda(self):

        if self.args.cuda == -1:
            device.isCuda = False
        else:
            device.isCuda = True

        if device.isCuda and not torch.cuda.is_available():
            print("CUDA not available on your machine. Setting it back to False")
            self.args.cuda = -1
            device.isCuda = False
