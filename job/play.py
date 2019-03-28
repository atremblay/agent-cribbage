from .job import Job
import gym
import gym_cribbage
from .register import register


@register
class Play(Job):
    def __init__(self, agent=None, args=None, logger=None):
        super().__init__(agent)
        super()._setup_job(__name__, args, logger)
        self.init_agent()

    def add_argument(self):
        # Add arguments
        pass  # No specific arguments

    def job(self, game_offset=0):

        games_data_files = []
        for game in range(self['number_games']):

            self.agents_reset(self.agents)
            env = gym.make('cribbage-v0')
            winner, hand, dealer = None, 0, None

            while winner is None:

                state, reward, done, debug = env.reset(dealer)
                self.logger.debug('Hand:' + str(hand))

                while not done:

                    self.logger.debug('\tCrib: ' + str(env.crib) + ' - Table_Count: ' + str(env.table_value))
                    if env.phase < 2:
                        card = self.agents[env.player].choose(state, env)
                        self.logger.debug('\t\tPhase: ' + str(env.phase) + ' Player:' + str(env.player) +
                                          ' chooses from: ' + str(state.hand) + ' -> ' + str(card))
                        state, reward, done, debug = env.step(card)

                    else:
                        state, reward, done, debug = env.step([])

                    self.logger.debug('\t\tPhase: ' + str(env.phase) + ' Player:' + str(state.reward_id) +
                                      ' gets reward: ' + str(reward))

                    self.agents[state.reward_id].store_reward(reward)
                    self.append_data(self.agents, env, state, hand, game)

                  # Check if current reward has determine a winner
                    if self.agents[state.reward_id].total_points >= 121:
                        winner = state.reward_id
                        done = True

                # Change dealer
                dealer = env.next_player(env.dealer)
                hand += 1

            self.agents[winner].data['winner'] = 1  # Store winner
            self.logger.debug('winner:' + str(winner))
            games_data_files.extend(self.dump_data(self.agents, game_offset+game))

        return games_data_files

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
                agent.append_data(hand, 0)
        elif env.phase != 0:
            agents[state.reward_id].append_data(hand, env.prev_phase)
        elif env.prev_phase == 2 and env.phase == 2:
            agents[state.reward_id].append_data(hand, env.prev_phase, no_state=True)

    def dump_data(self, agents, game):
        """
        Dump data in pickle at the end of the game.
        Each agent has a pickle.

        :param agents:
        :return:
        """
        data_files = []
        for agent_id, agent in enumerate(agents):
            data_files.append(agent.dump_data(self['save'], str(game)+'_'+str(agent_id)))

        return data_files