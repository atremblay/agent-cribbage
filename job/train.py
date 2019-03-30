from .job import Job
from utils.device import device
from .register import register
import os
from .play import Play
from algorithm.register import registry as algorithm_registry
from torch import nn
from torch.utils.data import DataLoader


@register
class Train(Job):
    def __init__(self):
        super().__init__()
        super()._setup_job(__name__, None, None)

        if self['data_dir'] is None:
            self['data_dir'] = '/'+os.path.join(*self['save'].split(os.path.sep)[:-2], 'job.play')

    def add_argument(self):
        # Add arguments
        self.parser.add_argument("--data_dir", default=None)
        self.parser.add_argument("--epochs", default=5, type=int)

    def init_algo(self):
        if self.args.algo == 'QLearning':
            return {}

    def get_data_files(self, agent_hash):
        for root, dirs, files in os.walk(self['data_dir']):
            for file in files:
                if file.endswith(agent_hash):
                    yield os.path.join(root, file)

    def job(self):

        game_offset = 0
        self.resolve_cuda()
        self.agents[0].init_optimizer()
        for epoch in range(self['epochs']):
            data_files = Play(agent=self.agents, args=self.args, logger=self.logger).job(game_offset=game_offset)
            game_offset += len(data_files)//len(self.agents)
            self.train(data_files, epoch)

        self.agents[0].save_checkpoint('./backup.tar', epoch)

    def train(self, data_files, epoch):

        training_contexts = self.init_training_contexts(self.agents[0], data_files)

        for context in training_contexts:

            nProcessed = 0
            for batch_idx, batch in enumerate(context['dataloader']):

                # Deformat data batch produced by Pytorch dataloader
                s_i, reward, s_prime = context['algorithm'].deformat(batch)

                # Device context
                s_i, reward, s_prime = [device(s) for s in s_i], device(reward), [device(s) for s in s_prime]

                if len(s_prime) > 0:
                    # Bootstrapping Value Evaluation
                    context['value_function'].eval()
                    reward += context['value_function'](*s_prime).flatten()

                # Current State evaluation and back propagation
                context['value_function'].train()
                output = context['value_function'](*s_i)
                context['optimizer'].zero_grad()
                loss = context['loss'](output, reward)
                loss.backward()
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

    @staticmethod
    def init_training_contexts(agent, data_files):

        training_contexts = []
        for i, value_function in enumerate(agent.value_functions):

            algorithm = algorithm_registry[agent.algorithms[i]['class']](data_files, **agent.algorithms[i]['kwargs'])
            loss = nn.MSELoss()

            for dataset in algorithm.datasets.values():
                context = {}
                dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
                context['optimizer'] = agent.optimizers[i]
                context['dataloader'] = dataloader
                context['value_function'] = value_function
                context['loss'] = loss
                context['algorithm'] = algorithm

                training_contexts.append(context)

        return training_contexts



