from .job import Job
import torch
import torch.optim as optim
from utils.device import device
from .register import register
import os
from agent.agent import Agent
import pickle
from .play import Play


@register
class Train(Job):
    def __init__(self):
        super().__init__()
        self.add_argument()
        super().setup_logging(__name__)
        if self['data_dir'] is None:
            self['data_dir'] = '/'+os.path.join(*self['save'].split(os.path.sep)[:-2], 'job.play')

    def add_argument(self):
        # Add arguments
        self.parser.add_argument('algo', choices=['QLearning'])
        self.parser.add_argument('--cuda', default=False, action='store_true')
        self.parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
        self.parser.add_argument("--lr", default=1e-3, type=float)
        self.parser.add_argument("--data_dir", default=None)
        self.parser.add_argument("--epochs", default=5, type=int)

    def get_algo_args(self):
        if self.args.algo == 'QLearning':
            return {}

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
            raise ValueError('Invalid optimizer value fro argument --opt:' + self.args.opt)

        return optimizer

    def get_data_files(self, agent_hash):
        for root, dirs, files in os.walk(self['data_dir']):
            for file in files:
                if file.endswith(agent_hash):
                    yield os.path.join(root, file)

    def job(self):
        agent_args = self.template_agent_args()
        agent = Agent(*agent_args)

        for epochs in range(self['epochs']):
            data_files = Play(agent=agent).job()
            for data_file in data_files:
                self.train(data_file)

    def train(self, data_file):
        test_data = pickle.load(open(data_file, 'rb'))
        self.agent.model.train(True)
        return
