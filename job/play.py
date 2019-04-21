from .job import Job
from .register import register
import gym
import gym_cribbage


@register
class Play(Job):
    def __init__(self, agent=None, args=None, logger=None):
        super().__init__(agent)
        super()._setup_job(__name__, args, logger)

    def add_argument(self):
        # Add arguments
        pass  # No specific arguments

    def job(self, game_offset=0):

        games_data_files = []
        game_statistics = [{'game_won': 0} for _ in self.agents]
        for game in range(self['number_games']):

            self.agents_reset(self.agents)
            env = gym.make('cribbage-v0')
            winner, hand, dealer = None, 0, None

            state, reward, done, debug = env.reset(dealer)

            while not done:

                # Resolve Previous player
                if state.reward_id is None:
                    previous_player = env.dealer
                else:
                    if state.hand_id != state.reward_id:
                        previous_player = state.reward_id

                if env.new_hand:
                    hand += 1

                if env.phase < 2:
                    card = self.agents[env.player].choose(state, env)
                    if env.phase == 1 or (env.phase == 0 and 'human' in self.agents[env.player].choose_phase_kwargs[0]):
                        self.logger.human('\t\t\t\t\t\t'*env.player+'Player '+str(env.player) +' plays: '+ str(card))

                    state, reward, done, debug = env.step(card)

                else:
                    self.logger.human('Starter: ' + str(env.starter))
                    self.logger.human('Hand Player '+str(env.player)+': ' + str(env.played[env.player]))
                    self.logger.human('Crib: ' + str(env.crib))
                    state, reward, done, debug = env.step([])

                # Update rewards
                self.agents[state.reward_id].store_reward(reward)
                self.agents[previous_player].update_offensive_defensive(hand, env.prev_phase, reward)

                self.logger.human('\t\t\t\t\t\t\t\t\t\t\t\t\tScore:' + str([a.total_points for a in self.agents]))
                self.append_data(self.agents, env, state, hand)

                # assert self.agents[0].total_points == env.scores[0]
                # assert self.agents[1].total_points == env.scores[1]

                # Check if current reward has determine a winner
                if self.agents[state.reward_id].total_points >= 121:
                    winner = state.reward_id

            self.agents[winner].data['winner'] = 1  # Store winner
            game_statistics[winner]['game_won'] += 1
            self.logger.debug('winner:' + str(winner))
            games_data_files.extend(self.dump_data(self.agents, game_offset+game))

        return games_data_files, game_statistics

    def agents_reset(self, agents):
        for agent in agents:
            agent.reset()

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