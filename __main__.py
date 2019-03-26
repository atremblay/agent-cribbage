import gym
import gym_cribbage
from agent.agent import Agent
import arguments
import logging


def agents_init(args):
    """ Initialize agents

    :param args:
    :return:
    """
    agents = []
    agent_args = [{'name': args['algo'], 'kwargs': args.get_algo_args()},
                  {'name': args['policy'], 'kwargs': args.get_policy_args()},
                  {
                   'name0': args['value_function0'], 'kwargs0': args.get_value_function_args()[0],
                   'name1': args['value_function1'], 'kwargs1': args.get_value_function_args()[1]
                   }]

    agent = Agent(*agent_args)

    agents.append(agent)
    for i in range(1, args['number_player']):
        agents.append(Agent(*agent_args))
        agents[i].value_function = agent.value_function  # All agents point to the same value function

    return agents


def append_data(agents, env, state, hand):
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


def dump_data(agents, winner, args):
    """
    Dump data in pickle at the end of the game.
    Each agent has a pickle.

    :param agents:
    :param winner:
    :return:
    """
    agents[winner].data['winner'] = 1  # Store winner
    for agent_id, agent in enumerate(agents):
        agent.dump_data(args['save'], agent_id)


if __name__ == "__main__":

    args = arguments.Args()
    args.save()
    logger = logging.getLogger(__name__)

    env = gym.make('cribbage-v0')

    agents = agents_init(args)
    winner, hand, dealer = None, 0, None
    while winner is None:

        state, reward, done, debug = env.reset(dealer)
        logger.info('Hand:' + str(hand))

        while not done:

            logger.info('\tCrib: ' + str(env.crib) + ' - Table_Count: '+str(env.table_value))
            if env.phase < 2:
                card = agents[env.player].choose(state, env)
                logger.info('\t\tPhase: ' + str(env.phase) + ' Player:' + str(env.player) +
                            ' chooses from: ' + str(state.hand) + ' -> ' + str(card))
                state, reward, done, debug = env.step(card)

            else:
                state, reward, done, debug = env.step([])

            logger.info('\t\tPhase: ' + str(env.phase) + ' Player:' + str(state.reward_id) +
                        ' gets reward: ' + str(reward))

            agents[state.reward_id].store_reward(reward)
            append_data(agents, env, state, hand)

            # Check if current reward has determine a winner
            if agents[state.reward_id].total_points >= 121:
                winner = state.reward_id
                done = True

        # Change dealer
        dealer = env.next_player(env.dealer)
        hand += 1

    logger.info('winner:' + str(winner))
    dump_data(agents, winner, args)
