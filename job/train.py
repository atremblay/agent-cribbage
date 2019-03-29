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
        self.init_agent()
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
        for epoch in range(self['epochs']):
            data_files = Play(agent=self.agents, args=self.args, logger=self.logger).job(game_offset=game_offset)
            game_offset += len(data_files)//len(self.agents)
            self.train(data_files, epoch)

        self.agents[0].save_checkpoint('./backup.tar', epoch)

    def train(self, data_files, epoch):
        agent = self.agents[0]
        dataset = algorithm_registry[agent.algorithms[0]['class']](data_files, **agent.algorithms[0]['kwargs'])
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        agent.value_functions[0].train(True)
        MSE = nn.MSELoss()
        nProcessed = 0
        agent.init_optimizer()

        for batch_idx, (inp, reward) in enumerate(dataloader):

            inp, reward = device(inp), device(reward)
            output = agent.value_functions[0](inp)
            agent.optimizers[0].zero_grad()
            loss = MSE(output, reward)
            loss.backward()
            agent.optimizers[0].step()

            # Statistics
            partialEpoch = epoch + batch_idx / len(dataloader)
            nProcessed += len(inp)
            self.logger.info(
                'Epoch: {:.2f} [{}/{} ({:.0f}%)], Loss: {:.6f}, Device: {}'.format(
                    partialEpoch, nProcessed, len(dataset), 100. * batch_idx / len(dataloader),
                    loss.item(), device)
            )


