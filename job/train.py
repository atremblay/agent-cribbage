from .job import Job
from ..utils.device import device
from .register import register
from .play import Play
from ..algorithm.register import registry as algorithm_registry
from torch import nn
from torch.utils.data import DataLoader
import os
import shutil
from datetime import datetime
import numpy as np
import copy
import torch

@register
class Train(Job):
    def __init__(self):
        super().__init__()
        super()._setup_job(__name__, None, None)

        # Initialize checkpoint directory and backup old dir
        if os.path.isdir(self.get_checkpoint_dir):
            shutil.make_archive(self.get_checkpoint_name+'_backup_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), 'zip',
                                self.get_checkpoint_dir)
            shutil.rmtree(self.get_checkpoint_dir)

        os.makedirs(self.get_checkpoint_dir)

    def add_argument(self):
        # Add arguments
        self.parser.add_argument("--data_dir", default=None)
        self.parser.add_argument("--epochs", default=5, type=int)
        self.parser.add_argument("--epoch_start", default=0, type=int)
        self.parser.add_argument("--dataepochs2keep", default=1, type=int)
        self.parser.add_argument("--checkpoint_period", default=10, type=int)
        self.parser.add_argument("--checkpoint_dir", default='./', type=str)
        self.parser.add_argument("--spin", default=1, type=int)
        self.parser.add_argument("--batchsize", default=64, type=int)
        self.parser.add_argument(
            "--patience",
            help="Number of epochs to wait before stopping the training",
            default=2,
            type=int
        )

    def job(self):

        game_offset = 0
        self.resolve_cuda()
        self.agents[0].init_optimizer()
        all_data_files = []
        for epoch in range(self['epoch_start'], self['epoch_start']+self['epochs']):
            data_files, _ = Play(agent=self.agents, args=self.args, logger=self.logger).job(game_offset=game_offset)
            game_offset += len(data_files)//len(self.agents)
            all_data_files.extend(data_files)

            # delete data files that are too old
            number_of_file_2_keep = self['dataepochs2keep']*self['number_games']*len(self.agents)
            [os.remove(f) for f in all_data_files[:-number_of_file_2_keep]]

            # Train on newest data files
            all_data_files = all_data_files[-number_of_file_2_keep:]
            self.train(all_data_files, epoch)

            if epoch % self['checkpoint_period'] == 0:
                self.agents[0].save_checkpoint(self.get_checkpoint_file(epoch), epoch)

            for a in self.agents:
                for s in a.scheduler:
                    if s is not None:
                        s.step()
                        self.logger.info(s.get_lr())

        self.agents[0].save_checkpoint(self.get_checkpoint_file(epoch), epoch)

    @property
    def get_checkpoint_dir(self):
        return os.path.join(self['checkpoint_dir'], self.get_checkpoint_name)

    def get_checkpoint_file(self, epoch):
        return os.path.join(self.get_checkpoint_dir, self.get_checkpoint_name+'_'+str(epoch)+'.tar')

    @property
    def get_checkpoint_name(self):
        return os.path.splitext(os.path.split(self['agent_yaml'])[1])[0]

    def train(self, data_files, epoch):

        training_contexts = self.init_training_contexts(self.agents[0], data_files)

        for context in training_contexts:

            old_value_function = copy.deepcopy(context['value_function'])
            old_value_function.eval()
            running_loss, best_loss, patience = 0, np.float('inf'), 0

            for i in range(self['spin']):

                if patience >= self['patience']:
                    break

                nProcessed = 0
                for batch_idx, (s_i, reward, s_primes) in enumerate(context['dataloader']):

                    # Device context
                    s_i, reward = [device(s) for s in s_i], device(reward)

                    if len(s_primes) > 0:
                        bootstrap_value = device(torch.zeros(len(s_primes), dtype=torch.float32))
                        for i_sample, (s_prime, idx_choice) in enumerate(s_primes):
                            # Bootstrapping Value Evaluation
                            if s_prime is not None:
                                s_prime = [device(s) for s in s_prime]
                                bootstrap_value[i_sample] = context['algorithm'].operator(old_value_function(*s_prime).flatten().detach(), idx_choice)

                        reward += bootstrap_value

                    # Current State evaluation and back propagation
                    context['value_function'].train()
                    output = context['value_function'](*s_i)
                    context['optimizer'].zero_grad()
                    loss = context['loss'](output.squeeze(), reward)
                    loss.backward()
                    running_loss += loss.item()
                    context['optimizer'].step()

                    # Statistics
                    partialEpoch = epoch + batch_idx / len(context['dataloader'])
                    nProcessed += len(reward)
                    self.logger.info(
                        'Epoch: {:.2f} [{}/{} ({:.0f}%)], Loss: {:.6f}, Device: {}'.format(
                            partialEpoch, nProcessed, len(context['dataloader'].dataset),
                            100. * batch_idx / len(context['dataloader']),
                            loss.item(), device)
                    )

                current_loss = running_loss / (batch_idx + 1)
                if current_loss > (best_loss - 0.0001):
                    patience += 1
                else:
                    patience = 0
                    best_loss = current_loss


    def init_training_contexts(self, agent, data_files):

        training_contexts = []
        for i, (value_function, policy) in enumerate(zip(agent.value_functions, agent.policies)):

            policy.step()  # Policy scheduler

            # If value function needs trainings
            if value_function.need_training:
                algorithm = algorithm_registry[agent.algorithms[i]['class']](data_files, value_function, policy, **agent.algorithms[i]['kwargs'])
                loss = nn.MSELoss()

                for dataset in algorithm.datasets.values():
                    context = {}
                    dataloader = DataLoader(dataset, batch_size=self['batchsize'], shuffle=True, num_workers=0, collate_fn=algorithm.collate_func)
                    context['optimizer'] = agent.optimizers[i]
                    context['scheduler'] = agent.scheduler[i]
                    context['dataloader'] = dataloader
                    context['value_function'] = value_function
                    context['loss'] = loss
                    context['algorithm'] = algorithm

                    training_contexts.append(context)

        return training_contexts

