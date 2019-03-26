import argparse
import torch
import torch.optim as optim
import logging
from utils.device import device
import os
import sys
import getpass
from logger_toolbox import setup_logging


class Args:

    def __init__(self):

        self.parser = argparse.ArgumentParser()
        # Add arguments
        self.parser.add_argument('algo', choices=['QLearning'])
        self.parser.add_argument('policy', choices=['Boltzmann', 'EpsilonGreedy', 'Random'])
        self.parser.add_argument('value_function0', choices=['FFW'])
        self.parser.add_argument('value_function1', choices=['LSTM', 'FFW'])
        self.parser.add_argument('--save', type=str, default='/home/execution')
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument("--tao", default=1, type=float)
        self.parser.add_argument("--epsilon", default=1e-3, type=float)
        self.parser.add_argument("--number_player", default=2, type=int)
        self.args = self.parser.parse_args()

        self.args.save = os.path.join(self.args.save, getpass.getuser(), 'agent-cribbage')
        setup_logging(self.args.save)
        # Append process Id that was created by logging
        self.args.save = os.path.join(self.args.save, 'pid' + str(os.getpid()))

    def __getitem__(self, item):
        return getattr(self.args, item)

    def get_policy_args(self):
        if self.args.policy == 'Boltzmann':
            return {'tao': self.args.tao}
        elif self.args.policy == 'EpsilonGreedy':
            return {'epsilon': self.args.epsilon}
        elif self.args.policy == 'Random':
            return {}

    def get_algo_args(self):
        if self.args.algo == 'QLearning':
            return {}

    def get_value_function_args(self):
        if self.args.value_function0 == 'FFW':
            VF0 = {}

        if self.args.value_function1 in ['LSTM', 'FFW']:
            VF1 = {}

        return VF0, VF1

    def resolve_cuda(self, net):

        device.isCuda = self.args.cuda

        if device.isCuda and not torch.cuda.is_available():
            print("CUDA not available on your machine. Setting it back to False")
            self.args.cuda = False
            device.isCuda = False

        if device.isCuda:
            net = net.cuda()

        return net

    def resolve_optimizer(self, net):

        if self.args.opt == 'sgd':
            optimizer = optim.SGD(
                net.parameters(),
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif self.args.opt == 'adam':
            optimizer = optim.Adam(net.parameters(), weight_decay=1e-4, lr=self.args.lr)
        elif self.args.opt == 'rmsprop':
            optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4, lr=self.args.lr)
        else:
            self.parser.print_help()
            raise ValueError( 'Invalid optimizer value fro argument --opt:' + self.args.opt)

        return optimizer

    def save(self):
        with open(os.path.join(self.args.save, 'commandline_args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

