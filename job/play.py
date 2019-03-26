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

    def job(self):

        env = gym.make('cribbage-v0')

        agents = self.agents_init()
        winner, hand, dealer = None, 0, None
        while winner is None:

            state, reward, done, debug = env.reset(dealer)
            self.logger.info('Hand:' + str(hand))

            while not done:

                self.logger.info('\tCrib: ' + str(env.crib) + ' - Table_Count: ' + str(env.table_value))
                if env.phase < 2:
                    card = agents[env.player].choose(state, env)
                    self.logger.info('\t\tPhase: ' + str(env.phase) + ' Player:' + str(env.player) +
                                     ' chooses from: ' + str(state.hand) + ' -> ' + str(card))
                    state, reward, done, debug = env.step(card)

                else:
                    state, reward, done, debug = env.step([])

                self.logger.info('\t\tPhase: ' + str(env.phase) + ' Player:' + str(state.reward_id) +
                                 ' gets reward: ' + str(reward))

                agents[state.reward_id].store_reward(reward)
                self.append_data(agents, env, state, hand)

                # Check if current reward has determine a winner
                if agents[state.reward_id].total_points >= 121:
                    winner = state.reward_id
                    done = True

            # Change dealer
            dealer = env.next_player(env.dealer)
            hand += 1

        self.logger.info('winner:' + str(winner))
        self.dump_data(agents, winner)

    def agents_init(self):
        """ Initialize agents

        :param args:
        :return:
        """
        agents = []
        agent_args = self.template_agent_args()
        agent = Agent(*agent_args)

        agents.append(agent)
        for i in range(1, self['number_player']):
            agents.append(Agent(*agent_args))
            agents[i].value_function = agent.value_function  # All agents point to the same value function

        return agents

    def append_data(self, agents, env, state, hand):
        """
        Append data to the buffer contextually to the phase

        :param agents:
        :param env:
        :return:
        """
        # At the end of phase 0, append data to buffer
        if env.prev_phase == 0 and env.phase == 1:
            for agent in agents:
                agent.append_data(hand, 0)
        elif env.phase != 0:
            agents[state.reward_id].append_data(hand, env.prev_phase)
        elif env.prev_phase == 2 and env.phase == 2:
            agents[state.reward_id].append_data(hand, env.prev_phase, no_state=True)

    def dump_data(self, agents, winner):
        """
        Dump data in pickle at the end of the game.
        Each agent has a pickle.

        :param agents:
        :param winner:
        :return:
        """
        agents[winner].data['winner'] = 1  # Store winner
        for agent_id, agent in enumerate(agents):
            agent.dump_data(self['save'], agent_id)