from .job import Job
from .register import register
from .play import Play
import os
import matplotlib.pyplot as plt


@register
class Evaluate(Job):
    def __init__(self):
        super().__init__(None)
        super()._setup_job(__name__, None, None)

    def add_argument(self):
        self.parser.add_argument("--config_dir", default=None)
        self.parser.add_argument("--model_dir", default=None)

    @staticmethod
    def get_files(directory, ext):
        files = []
        for f in os.listdir(directory):
            if f.endswith(ext):
                files.append(os.path.join(directory, f))
        return files

    def job(self):
        models = self.get_files(self['model_dir'], '.tar')
        configs = self.get_files(self['config_dir'], '.yaml')

        for config in configs:
            self.logger.info(config)
            # Init agents for this config
            self.agents = None
            self.init_agent(config)

            config_statistics = {}
            # Loop over all models
            for model in sorted(models):
                self.logger.info('\t'+model)
                self.agents[0].load_checkpoint(model)
                _, statistics = Play(agent=self.agents, args=self.args, logger=self.logger).job()

                self.logger.info('\t\t'+str(statistics[0]['game_won']))
                config_statistics[int(model.split('_')[-1].split('.')[0])] = statistics

            self.plot_agent_battles(config_statistics, config)

    def plot_agent_battles(self, config_statistics, config):

        plt.clf()

        x = [k for k in config_statistics.keys()]
        x.sort()

        y = [config_statistics[k][0]['game_won'] for k in x]

        name_config = os.path.splitext(os.path.split(config)[-1])[0]
        plt.plot(x, y)
        plt.title(name_config)
        plt.xlabel('Epochs')
        plt.ylabel('Agent Wins over '+str(self['number_games'])+' games')
        plt.savefig(os.path.join(self['model_dir'], name_config+'.pdf'))
