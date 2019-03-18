import gym
import gym_cribbage
from agent.agent import Agent
import arguments

if __name__ == "__main__":

    args = arguments.Args()
    args.save()

    env = gym.make('cribbage-v0')
    agent = Agent({'name': args['algo'], 'kwargs': args.get_algo_args()},
                  {'name': args['policy'], 'kwargs': args.get_policy_args()})

    args.logger.info('salut')