from .job import Job
import gym
import gym_cribbage
from agent.agent import Agent
from .register import register


@register
class Play(Job):
    def __init__(self):
        super().__init__()
        self.add_argument()
        super().setup_logging(__name__)

    def add_argument(self):
        # Add arguments
        self.parser.add_argument("--number_player", default=2, type=int)
        self.parser.add_argument("--number_games", default=1, type=int)

    def job(self):

        agents = self.agents_init()

        for game in range(self['number_games']):

            self.agents_reset(agents)
            env = gym.make('cribbage-v0')
            winner, hand, dealer = None, 0, None

            while winner is None:

                state, reward, done, debug = env.reset(dealer)
                self.logger.debug('Hand:' + str(hand))

                while not done:

                    self.logger.debug('\tCrib: ' + str(env.crib) + ' - Table_Count: ' + str(env.table_value))
                    if env.phase < 2:
                        card = agents[env.player].choose(state, env)
                        self.logger.debug('\t\tPhase: ' + str(env.phase) + ' Player:' + str(env.player) +
                                          ' chooses from: ' + str(state.hand) + ' -> ' + str(card))
                        state, reward, done, debug = env.step(card)

                    else:
                        state, reward, done, debug = env.step([])

                    self.logger.debug('\t\tPhase: ' + str(env.phase) + ' Player:' + str(state.reward_id) +
                                      ' gets reward: ' + str(reward))

                    agents[state.reward_id].store_reward(reward)
                    self.append_data(agents, env, state, hand, game)

                    # Check if current reward has determine a winner
                    if agents[state.reward_id].total_points >= 121:
                        winner = state.reward_id
                        done = True

                # Change dealer
                dealer = env.next_player(env.dealer)
                hand += 1

            agents[winner].data[game]['winner'] = 1  # Store winner
            self.logger.debug('winner:' + str(winner))
        self.dump_data(agents)

    def agents_init(self):
        """ Initialize agents

        :param args:
        :return:
        """
        agents = []
        agent_args = self.template_agent_args()
        agent = Agent(*agent_args, number_games=self['number_games'])

        agents.append(agent)
        for i in range(1, self['number_player']):
            agents.append(Agent(*agent_args, number_games=self['number_games']))
            agents[i].value_function = agent.value_function  # All agents point to the same value function

        return agents

    def agents_reset(self, agents):
        for agent in agents:
            agent.reset()

    def append_data(self, agents, env, state, hand, game):
        """
        Append data to the buffer contextually to the phase

        :param agents:
        :param env:
        :return:
        """
        # At the end of phase 0, append data to buffer
        if env.prev_phase == 0 and env.phase == 1:
            for agent in agents:
                agent.append_data(game, hand, 0)
        elif env.phase != 0:
            agents[state.reward_id].append_data(game, hand, env.prev_phase)
        elif env.prev_phase == 2 and env.phase == 2:
            agents[state.reward_id].append_data(game, hand, env.prev_phase, no_state=True)

    def dump_data(self, agents):
        """
        Dump data in pickle at the end of the game.
        Each agent has a pickle.

        :param agents:
        :return:
        """
        for agent_id, agent in enumerate(agents):
            agent.dump_data(self['save'], agent_id)