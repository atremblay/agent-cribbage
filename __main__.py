import gym
import gym_cribbage
from agent.agent import Agent
import arguments


def agents_init(args):
    """ Initialize agents

    :param args:
    :return:
    """
    agents = []
    agent_args = [{'name': args['algo'], 'kwargs': args.get_algo_args()},
                  {'name': args['policy'], 'kwargs': args.get_policy_args()},
                  {'name': args['value_function'], 'kwargs': args.get_value_function_args()}]

    agent = Agent(*agent_args)

    agents.append(agent)
    for i in range(1, args['number_player']):
        agents.append(Agent(*agent_args))
        agents[i].value_function = agent.value_function  # All agents point to the same value function

    return agents


if __name__ == "__main__":

    args = arguments.Args()
    args.save()

    env = gym.make('cribbage-v0')

    agents = agents_init(args)
    winner, hand = None, 0
    while winner is None:
        state, reward, done, debug = env.reset()
        hand += 1
        args.logger.info('Hand:' + str(hand))
        while not done:

            args.logger.info('\tCrib: ' + str(env.crib)+ ' - Table_Count: '+str(env.table_value))
            if env.phase < 2:
                idx_action = agents[env.player].policy.choose(state.hand)
                args.logger.info('\t\tPhase: ' + str(env.phase) + ' Player:' + str(env.player) +
                                 ' chooses from: ' + str(state.hand) + ' -> ' + str(state.hand[idx_action]))
                state, reward, done, debug = env.step(state.hand[idx_action])

            else:
                state, reward, done, debug = env.step([])

            args.logger.info('\t\tPhase: ' + str(env.phase) + ' Player:' + str(env.last_player) +
                             ' gets reward: ' + str(reward))

            agents[env.last_player].reward.append(reward)

            if agents[env.last_player].total_points >= 121:
                winner = env.last_player
                done = True

    args.logger.info('winner:' + str(winner))